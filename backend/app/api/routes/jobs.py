import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.agents.graph import get_graph
from app.agents.state import AgentState
from app.core.config import settings
from app.core.database import get_db
from app.core.llm_factory import validate_llm_config
from app.models.job import Job
from app.models.schemas import CreateJobRequest

router = APIRouter()

# In-memory queues for SSE: job_id -> asyncio.Queue
_job_queues: dict[str, asyncio.Queue] = {}


def _get_or_create_queue(job_id: str) -> asyncio.Queue:
    if job_id not in _job_queues:
        _job_queues[job_id] = asyncio.Queue(maxsize=1000)
    return _job_queues[job_id]


async def _run_pipeline(
    job_id: str,
    dataset_path: str,
    dataset_name: str,
    user_description: str,
    target_column: Optional[str],
    llm_config: dict,
):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    AsyncSess = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    queue = _get_or_create_queue(job_id)

    async def push_event(event_type: str, data: dict):
        try:
            await queue.put({"event": event_type, "data": data})
        except asyncio.QueueFull:
            pass

    initial_state: AgentState = {
        "llm_config": llm_config,
        "job_id": job_id,
        "dataset_path": dataset_path,
        "user_description": user_description,
        "target_column": target_column,
        "problem_type": "unknown",
        "profile": None,
        "df_sample_path": None,
        "original_columns": [],
        "eda_findings": [],
        "eda_charts": [],
        "target_analysis": None,
        "eda_narrative": None,
        "features_applied": [],
        "processed_dataset_path": None,
        "preprocessor_path": None,
        "feature_names": [],
        "dropped_columns": [],
        "plan": None,
        "experiments": [],
        "baseline_score": None,
        "critic_history": [],
        "retry_count": 0,
        "best_model_name": None,
        "best_model_path": None,
        "evaluation_metrics": None,
        "evaluation_charts": [],
        "model_explanation": None,
        "statistical_comparison": None,
        "shap_summary": None,
        "report_pdf_path": None,
        "report_json": None,
        "current_agent": "starting",
        "status": "running",
        "progress_pct": 0,
        "messages": [],
        "errors": [],
        "warnings": [],
        "token_usage": [],
        "trace_events": [],
    }

    try:
        graph = get_graph()
        config = {"configurable": {"thread_id": job_id}}
        last_progress = 0
        seen_messages = 0
        seen_trace_events = 0

        # Accumulate final state from stream chunks (avoids get_state() which needs checkpointer)
        final: dict = {}

        async for chunk in graph.astream(initial_state, config=config):
            for node_name, node_state in chunk.items():
                if not isinstance(node_state, dict):
                    continue

                # Merge into running final state
                final.update(node_state)

                current_agent = node_state.get("current_agent", node_name)
                progress = node_state.get("progress_pct", last_progress)
                messages = node_state.get("messages", [])

                # Push new messages to SSE queue
                for msg in messages[seen_messages:]:
                    await push_event("update", {
                        "agent": current_agent,
                        "progress_pct": progress,
                        "message": msg,
                    })
                seen_messages = len(messages)

                # Push richer structured events (Planner's plan, Critic's verdict, retry
                # loop-backs, token usage) — this is what drives the live agent-brain UI,
                # separate from the plain-text human log above.
                trace_events = node_state.get("trace_events", [])
                for evt in trace_events[seen_trace_events:]:
                    await push_event(evt["kind"], {
                        "agent": evt["agent"],
                        "payload": evt["payload"],
                        "ts": evt["ts"],
                    })
                seen_trace_events = len(trace_events)
                last_progress = progress

                # Update DB
                async with AsyncSess() as session:
                    job = await session.get(Job, job_id)
                    if job:
                        job.status = node_state.get("status", "running")
                        job.progress_pct = progress
                        job.current_agent = current_agent
                        job.messages = messages
                        job.errors = node_state.get("errors", [])
                        job.warnings = node_state.get("warnings", [])
                        if node_state.get("best_model_name"):
                            job.best_model_name = node_state["best_model_name"]
                        if node_state.get("problem_type"):
                            job.problem_type = node_state["problem_type"]
                        if node_state.get("report_json"):
                            job.report_json = node_state["report_json"]
                        if node_state.get("report_pdf_path"):
                            job.report_pdf_path = node_state["report_pdf_path"]
                        # Compute best score
                        exps = node_state.get("experiments", [])
                        if exps:
                            completed = [e for e in exps if e.get("status") in ("completed", "completed_reduced")]
                            if completed:
                                job.best_model_score = max(e.get("primary_score", 0) for e in completed)
                        await session.commit()

        status = final.get("status", "completed")

        async with AsyncSess() as session:
            job = await session.get(Job, job_id)
            if job:
                job.status = status
                job.progress_pct = 100 if status == "completed" else job.progress_pct
                if final.get("report_json"):
                    job.report_json = final["report_json"]
                if final.get("report_pdf_path"):
                    job.report_pdf_path = final["report_pdf_path"]
                await session.commit()

        if status == "completed":
            await push_event("completed", {
                "report_url": f"/api/reports/{job_id}/pdf",
                "job_id": job_id,
            })
        else:
            errors = final.get("errors", [])
            await push_event("failed", {"error": "; ".join(errors) if errors else "Pipeline failed"})

    except Exception as e:
        error_msg = str(e)
        if llm_config.get("api_key"):
            error_msg = error_msg.replace(llm_config["api_key"], "[REDACTED]")

        async with AsyncSess() as session:
            job = await session.get(Job, job_id)
            if job:
                job.status = "failed"
                job.errors = [f"Pipeline error: {error_msg[:200]}"]
                await session.commit()

        await push_event("failed", {"error": f"Pipeline error: {error_msg[:200]}"})
    finally:
        # Signal stream end
        await push_event("done", {})
        await engine.dispose()


@router.post("/validate-key")
async def validate_key(request: Request):
    try:
        llm_config = request.state.llm_config
        await asyncio.to_thread(validate_llm_config, llm_config)
        return {"valid": True, "provider": llm_config["provider"]}
    except ValueError as e:
        error_msg = str(e)
        if request.state.llm_config.get("api_key"):
            error_msg = error_msg.replace(request.state.llm_config["api_key"], "[REDACTED]")
        return {"valid": False, "error": error_msg}
    except Exception:
        return {"valid": False, "error": "Validation failed — check your key and try again."}


@router.post("/jobs")
async def create_job(
    payload: CreateJobRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    llm_config = request.state.llm_config

    # Resolve dataset path
    if payload.dataset_source == "upload":
        if not payload.uploaded_file_path:
            raise HTTPException(status_code=400, detail="uploaded_file_path required for upload source")
        if not os.path.exists(payload.uploaded_file_path):
            raise HTTPException(status_code=404, detail="Uploaded file not found")
        dataset_path = payload.uploaded_file_path
        dataset_name = Path(payload.uploaded_file_path).name

    elif payload.dataset_source == "huggingface":
        if not payload.hf_dataset_id:
            raise HTTPException(status_code=400, detail="hf_dataset_id required")
        try:
            from datasets import load_dataset
            ds = load_dataset(payload.hf_dataset_id, split=payload.hf_split)
            df = ds.to_pandas()
            if len(df) > payload.hf_max_rows:
                df = df.sample(n=payload.hf_max_rows, random_state=42)
            os.makedirs(settings.STORAGE_PATH, exist_ok=True)
            fname = f"{uuid.uuid4().hex}_hf_{payload.hf_dataset_id.replace('/', '_')}.parquet"
            dataset_path = os.path.join(settings.STORAGE_PATH, fname)
            df.to_parquet(dataset_path, index=False)
            dataset_name = f"HuggingFace: {payload.hf_dataset_id}"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load HuggingFace dataset: {str(e)[:200]}")

    elif payload.dataset_source == "kaggle":
        if not payload.kaggle_dataset or not payload.kaggle_filename:
            raise HTTPException(status_code=400, detail="kaggle_dataset and kaggle_filename required")
        try:
            import kaggle
            os.makedirs(settings.STORAGE_PATH, exist_ok=True)
            kaggle.api.dataset_download_file(
                payload.kaggle_dataset, payload.kaggle_filename,
                path=settings.STORAGE_PATH
            )
            dataset_path = os.path.join(settings.STORAGE_PATH, payload.kaggle_filename)
            dataset_name = f"Kaggle: {payload.kaggle_dataset}/{payload.kaggle_filename}"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load Kaggle dataset: {str(e)[:200]}")
    else:
        raise HTTPException(status_code=400, detail=f"Unknown dataset_source: {payload.dataset_source}")

    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        status="queued",
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        user_description=payload.user_description,
        target_column=payload.target_column,
        messages=[],
        errors=[],
        warnings=[],
    )
    db.add(job)
    await db.commit()

    # Pre-create queue in the current event loop before background task runs
    _get_or_create_queue(job_id)

    background_tasks.add_task(
        _run_pipeline,
        job_id=job_id,
        dataset_path=dataset_path,
        dataset_name=dataset_name,
        user_description=payload.user_description,
        target_column=payload.target_column,
        llm_config=llm_config,
    )

    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}/stream")
async def stream_job(job_id: str, request: Request):
    queue = _get_or_create_queue(job_id)

    async def event_generator():
        retries = 0
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield {"event": event["event"], "data": json.dumps(event["data"])}
                if event["event"] in ("completed", "failed", "done"):
                    break
            except asyncio.TimeoutError:
                yield {"event": "ping", "data": json.dumps({"t": "ping"})}
                retries += 1
                if retries > 60:
                    break
            except Exception:
                break

    return EventSourceResponse(event_generator())


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress_pct": job.progress_pct,
        "current_agent": job.current_agent,
        "dataset_name": job.dataset_name,
        "user_description": job.user_description,
        "target_column": job.target_column,
        "problem_type": job.problem_type,
        "best_model_name": job.best_model_name,
        "best_model_score": job.best_model_score,
        "messages": job.messages or [],
        "errors": job.errors or [],
        "warnings": job.warnings or [],
        "report_json": job.report_json,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


@router.get("/jobs")
async def list_jobs(db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(Job).order_by(Job.created_at.desc()).limit(50))
    jobs = result.scalars().all()
    return [
        {
            "job_id": j.id,
            "status": j.status,
            "progress_pct": j.progress_pct,
            "dataset_name": j.dataset_name,
            "problem_type": j.problem_type,
            "best_model_score": j.best_model_score,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]
