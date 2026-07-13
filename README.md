# ML Intern — A Self-Correcting Multi-Agent Data Scientist

Upload a tabular dataset, describe your goal, and a team of LLM agents plans a strategy, engineers
features, trains models, **critiques its own results, and retries when it isn't confident** —
before writing a report with statistical significance testing and SHAP explainability, not just
a bare accuracy number.

![Demo placeholder — replace with GIF](docs/demo.gif)

## Why this is different from a typical "AutoML + LLM" project

Most "autonomous data science agent" projects are a fixed sklearn pipeline with an LLM bolted on
to narrate the results. This one isn't:

1. **A real agentic loop, not a linear pipeline.** A `Planner` agent decides feature-drop and
   model-shortlist strategy from the data's actual characteristics (not a hardcoded "always train
   these 5 models"). A `Critic` agent then reviews the results and can send the run **back** to
   feature engineering or model training — a bounded self-correction loop, not just "if error,
   stop."
2. **Statistical rigor, not a bare number.** Every run trains a naive baseline alongside the real
   models and reports a bootstrapped confidence interval and a probability-of-superiority against
   that baseline — *"14% better than guessing, 95% CI 9–19%, high confidence"* instead of
   `accuracy=0.87`.
3. **Real explainability.** SHAP feature attributions (not just `feature_importances_`) for the
   final model, with a per-instance breakdown of what drove one specific prediction.
4. **A live view of the agents actually thinking.** The frontend streams an animated graph of the
   agent pipeline (built on the same graph library patterns used by LangGraph Studio) — nodes
   light up as they run, and the Critic's retry loop-back visibly fires and reroutes execution
   when it happens. A console panel shows the Planner's reasoning and the Critic's verdicts live,
   not just a progress bar.
5. **Agent evals in CI**, not just unit tests. `backend/tests/evals/` runs the Planner and Critic
   against synthetic datasets with planted issues (a leakage column, a no-signal dataset) and
   asserts they actually catch them — plus an opt-in eval against a *real* LLM to check genuine
   decision quality, not just wiring.
6. **Full observability.** Token usage, cost, and latency are tracked per LLM call and shown in
   the report — not hidden.

The base is still a legitimate FastAPI + LangGraph + Next.js system with real cross-validation,
timeout/OOM handling, and MLflow tracking — the agentic layer above is what makes it worth a
second look.

## Architecture

```
User Browser
│
├─ API Key (localStorage only, never leaves browser, sent only in headers — never in a URL)
│
▼
Next.js Frontend ──── SSE live trace (thoughts / plan / critic verdicts / retries / tokens) ────► FastAPI Backend
                                                                                                          │
                                                                                                 LangGraph Agent Pipeline
                                                                            ┌─────────────────────────────────────────────────────┐
                                                                            │  EDA ─► Planner ─► Features ─► Train ─► Critic       │
                                                                            │                       ▲                    │         │
                                                                            │                       └──── retry (≤2) ────┘         │
                                                                            │                                            │         │
                                                                            │                                       Evaluation     │
                                                                            │                                            │         │
                                                                            │                                         Report       │
                                                                            └─────────────────────────────────────────────────────┘
                                                                                                          │
                                                                                             MLflow Tracking · SHAP · Bootstrap CI
                                                                                             PDF Report (ReportLab)
                                                                                             SQLite (dev) / Postgres (prod)
```

**Planner** reads the EDA findings and decides: which columns to drop (acting on leakage/ID
columns EDA flagged, not just reporting them), and which model shortlist makes sense for this
dataset's size and shape. **Critic** reviews training results against the baseline and can route
back to `Features` or `Train` with a revised strategy (bounded to 2 retries, then it force-approves
with an honest low-confidence caveat rather than looping forever).

## Quick Start

```bash
git clone https://github.com/your-user/ml-intern
cd ml-intern
cp .env.example .env
make dev
```

Open http://localhost:3000, enter your API key, and upload a dataset.

## BYOK — Bring Your Own Key

**This app costs $0 to operate.** You provide your own LLM API key. It is:

- Stored only in your browser (`localStorage`)
- Sent only in request headers — including the live event stream (no API keys in URLs/query
  strings, which would otherwise leak into server logs and browser history)
- Automatically redacted from any error messages

### Supported providers

| Provider | Free tier? | Recommended model |
|---|---|---|
| **Anthropic** ← recommended | $5 free on signup | Claude Haiku 4.5 |
| OpenAI | No | GPT-4o Mini |
| **Groq** | ✅ Yes, no card | Llama 3.3 70B |

## Supported Dataset Formats

- CSV (any encoding — auto-detected)
- Parquet
- Excel (`.xlsx`, `.xls`)
- JSON
- TSV
- ZIP containing a CSV

**Dataset sources:** Direct file upload, HuggingFace Hub (`scikit-learn/iris`), Kaggle (requires API key)

## What ML Intern Does

1. **EDA Agent** — Profiles columns, detects target, finds missing values, leakage, class imbalance. Generates 7 charts. LLM writes narrative.
2. **Planner Agent** — Decides which columns to drop (leakage/ID) and which model shortlist to train, based on the actual dataset shape — an LLM decision, not a hardcoded list.
3. **Feature Agent** — Applies the Planner's drop decisions, imputes missing, extracts datetime features, caps outliers, encodes categoricals, scales numerics.
4. **Experiment Agent** — Trains the Planner's model shortlist plus a naive baseline, with cross-validation. Tracks in MLflow.
5. **Critic Agent** — Reviews the best model vs. the baseline, flags instability or insufficient signal, and can send the run back to Features or Train with a revised strategy (bounded to 2 retries).
6. **Evaluation Agent** — Retrains the approved model, computes final metrics, bootstrapped confidence intervals, baseline comparison, SHAP explainability, and diagnostic charts.
7. **Report Agent** — Generates a multi-page PDF with all findings, charts, statistical confidence, agent self-review history, and LLM-written recommendations.

## Agent Evals

`backend/tests/evals/` checks the agents' actual decisions, not just that the code runs:

- Planner correctly drops a planted leakage column (both via the LLM and via the deterministic
  fallback path that reads EDA's own leakage findings).
- Critic flags `insufficient_signal` when the best model barely beats baseline, and never loops
  past its hard retry cap even if the LLM keeps asking to retry.
- An opt-in `@pytest.mark.live_llm` suite (skipped by default) runs the same checks against a
  real configured model — set `RUN_LIVE_LLM_EVALS=1` plus `TEST_LLM_API_KEY` (and optionally
  `TEST_LLM_PROVIDER` / `TEST_LLM_MODEL`) to run it locally, or trigger the `backend-live-llm-evals`
  job manually in CI.

## Deployment

### Railway (backend) + Vercel (frontend)

```bash
# Backend on Railway
railway init
railway up --service backend

# Frontend on Vercel
vercel --prod

# Set env vars:
NEXT_PUBLIC_API_URL=https://your-backend.railway.app
```

### Docker (self-hosted)

```bash
make prod
```

## Adding a New Model

Add to `CLASSIFICATION_MODELS` or `REGRESSION_MODELS` in `backend/app/tools/ml_tools.py`:

```python
"my_model": lambda: MyClassifier(param=value, random_state=42, n_jobs=-1),
```

The Planner will consider it automatically the next time it builds a shortlist.

## Known Limitations

- NLP tasks (text classification) are not supported — only tabular datasets
- Maximum file upload size: 500MB
- LLM quality of planning/critique/narrative depends on the user's chosen model (Haiku is fast but less insightful than Sonnet)
- MLflow tracking requires the MLflow server to be running (gracefully skipped if unavailable)
- Very wide datasets (500+ columns) are auto-reduced to top 200 by variance
- The Critic's retry loop is bounded to 2 attempts — after that it approves with a low-confidence caveat rather than looping indefinitely
- `mypy` is run in CI as informational (non-blocking) — there is pre-existing type debt around SQLAlchemy's declarative Column typing and a couple of third-party stub mismatches (langchain, aiofiles) that don't affect runtime behavior
- **Concurrency under load**: the pipeline runs as a FastAPI `BackgroundTask` on a single-process `uvicorn` server. LLM calls are offloaded to worker threads (`asyncio.to_thread`) and model training already ran through a `ThreadPoolExecutor`, which fixed an actual crash (the server's socket-accept loop died under prolonged blocking during live testing). However, CPU-bound scikit-learn/pandas work still holds Python's GIL for large stretches, so the server can become slow to respond to *other* requests (health checks, a second job) while one job is mid-training — confirmed via live testing, not theoretical. For a single-user BYOK demo this is a non-issue; for multi-tenant production use, the correct fix is moving model training to a `ProcessPoolExecutor` (or a separate worker process/queue like Celery/RQ) so it's immune to the GIL — noted here rather than rushed in, since a live-tested partial fix is more trustworthy than an unverified "complete" one.

## Tech Stack

**Backend:** FastAPI, LangGraph (with a bounded reflection loop), LangChain (structured tool-calling
output), scikit-learn, XGBoost, LightGBM, SHAP, ReportLab, MLflow, SQLAlchemy, SQLite
**Frontend:** Next.js 14, TypeScript, Tailwind CSS, Recharts, React Flow (live agent graph)
**Observability:** Per-call token/cost/latency tracking, structured SSE trace events
**Testing:** pytest, agent evals (deterministic + opt-in live-LLM), GitHub Actions CI
**Infrastructure:** Docker Compose, Uvicorn

## License

MIT
