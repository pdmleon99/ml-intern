import asyncio
import concurrent.futures
import copy
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from app.core.config import settings
from app.tools.ml_tools import CLASSIFICATION_MODELS, REGRESSION_MODELS

from .state import AgentState


async def run_experiment_agent(state: AgentState) -> AgentState:
    state["current_agent"] = "experiments"
    state["progress_pct"] = 42
    state["messages"].append("🏋️ Training models...")

    try:
        if not state.get("processed_dataset_path") or not state.get("preprocessor_path"):
            state["errors"].append("No processed dataset or preprocessor found.")
            state["status"] = "failed"
            return state

        df = pd.read_parquet(state["processed_dataset_path"])
        target = state["target_column"]
        X = df.drop(columns=[target])
        y = df[target]

        # Load the unfitted preprocessor (cross_validate clones it per fold)
        preprocessor = joblib.load(state["preprocessor_path"])

        # Encode string target for classification
        if state["problem_type"] == "classification":
            if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

        n_classes = y.nunique() if state["problem_type"] == "classification" else None
        if state["problem_type"] == "classification" and n_classes <= 1:
            state["errors"].append("Target has only one class — cannot train a classifier.")
            state["status"] = "failed"
            return state

        n_rows = len(df)
        n_splits = 3 if n_rows < 500 else 5

        if state["problem_type"] == "classification":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            is_binary = n_classes == 2
            if is_binary:
                scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted", "roc_auc": "roc_auc"}
                primary_metric = "roc_auc"
            else:
                scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted",
                           "roc_auc": "roc_auc_ovr_weighted"}
                primary_metric = "f1_weighted"
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            primary_metric = "r2"
            scoring = {"r2": "r2", "neg_mae": "neg_mean_absolute_error",
                       "neg_rmse": "neg_root_mean_squared_error"}

        models = (CLASSIFICATION_MODELS if state["problem_type"] == "classification"
                  else REGRESSION_MODELS)

        # Planner decides which models are worth training for this dataset (agentic
        # selection, not always-train-all-5). Falls back to every model if the plan
        # is missing/empty or names nothing valid.
        shortlist = (state.get("plan") or {}).get("model_shortlist") or []
        shortlist = [m for m in shortlist if m in models]
        models_to_run = {name: models[name] for name in shortlist} if shortlist else dict(models)

        # Always add a naive baseline — needed to judge whether the real models add
        # any value at all (kills "bare accuracy number" reporting).
        if state["problem_type"] == "classification":
            from sklearn.dummy import DummyClassifier
            models_to_run["baseline_dummy"] = lambda: DummyClassifier(
                strategy="stratified", random_state=42
            )
        else:
            from sklearn.dummy import DummyRegressor
            models_to_run["baseline_dummy"] = lambda: DummyRegressor(strategy="mean")

        experiments = []

        mlflow_available = False
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(f"ml-intern-{state['job_id'][:8]}")
            mlflow_available = True
        except Exception:
            state["warnings"].append("MLflow server unavailable — skipping experiment tracking")

        # Subsample for very large datasets (pipeline does the rest)
        X_cv, y_cv = X, y
        if n_rows > 50_000:
            idx = np.random.RandomState(42).choice(n_rows, 50_000, replace=False)
            X_cv = X.iloc[idx]
            y_cv = y.iloc[idx]

        n_models = len(models_to_run)
        for i, (name, model_factory) in enumerate(models_to_run.items()):
            state["messages"].append(f"  Training {name}...")
            start = time.time()

            # Each model gets its own pipeline (fresh preprocessor copy)
            pipeline = Pipeline([
                ("preprocessor", copy.deepcopy(preprocessor)),
                ("model", model_factory()),
            ])

            try:
                # Trains in a worker thread and awaits it — `cross_validate` blocks for
                # seconds-to-minutes; awaiting a wrapped future (instead of the old
                # `future.result(timeout=...)`, which blocks the event loop directly) keeps
                # the server responsive to other requests (health checks, other jobs' SSE
                # streams) while this model trains.
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = executor.submit(
                    cross_validate,
                    pipeline,
                    X_cv,
                    y_cv,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=1,
                    error_score="raise",
                )
                try:
                    cv_results = await asyncio.wait_for(
                        asyncio.wrap_future(future), timeout=settings.MAX_TRAINING_TIMEOUT_SECONDS
                    )
                finally:
                    executor.shutdown(wait=False)

                elapsed = time.time() - start
                cv_scores = {
                    k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    for k, v in cv_results.items()
                    if k.startswith("test_")
                }
                pm_key = f"test_{primary_metric}"
                primary_score = cv_scores.get(pm_key, {}).get("mean", 0.0)

                exp_record = {
                    "model_name": name,
                    "cv_scores": cv_scores,
                    "primary_metric": primary_metric,
                    "primary_score": float(primary_score),
                    # raw per-fold scores — needed for the paired bootstrap comparison against
                    # the baseline in evaluation_agent (mean/std alone can't support that test)
                    "primary_score_folds": [float(v) for v in cv_results.get(pm_key, [])],
                    "train_time_s": round(elapsed, 2),
                    "status": "completed",
                    "mlflow_run_id": None,
                    "is_baseline": name == "baseline_dummy",
                }

                if mlflow_available:
                    try:
                        with mlflow.start_run(run_name=name):
                            for mn, ms in cv_scores.items():
                                mlflow.log_metric(mn.replace("test_", "") + "_mean", ms["mean"])
                            mlflow.log_param("model", name)
                            mlflow.log_param("n_splits", n_splits)
                            exp_record["mlflow_run_id"] = mlflow.active_run().info.run_id
                    except Exception:
                        pass

                experiments.append(exp_record)
                state["messages"].append(
                    f"  ✓ {name}: {primary_metric}={primary_score:.4f} ({elapsed:.1f}s)"
                )

            except concurrent.futures.TimeoutError:
                experiments.append({
                    "model_name": name, "status": "timed_out",
                    "primary_score": 0, "train_time_s": settings.MAX_TRAINING_TIMEOUT_SECONDS,
                    "primary_metric": primary_metric,
                })
                state["warnings"].append(f"{name} timed out after {settings.MAX_TRAINING_TIMEOUT_SECONDS}s")

            except MemoryError:
                try:
                    X_red = X_cv.sample(frac=0.5, random_state=42)
                    y_red = y_cv.loc[X_red.index]
                    pipeline2 = Pipeline([
                        ("preprocessor", copy.deepcopy(preprocessor)),
                        ("model", model_factory()),
                    ])
                    cv_results = await asyncio.to_thread(
                        cross_validate, pipeline2, X_red, y_red, cv=cv, scoring=scoring, n_jobs=1
                    )
                    elapsed = time.time() - start
                    cv_scores = {
                        k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                        for k, v in cv_results.items()
                        if k.startswith("test_")
                    }
                    experiments.append({
                        "model_name": name,
                        "cv_scores": cv_scores,
                        "primary_metric": primary_metric,
                        "primary_score": float(cv_scores.get(f"test_{primary_metric}", {}).get("mean", 0)),
                        "train_time_s": round(elapsed, 2),
                        "status": "completed_reduced",
                        "note": "trained on 50% sample due to memory constraints",
                    })
                    state["warnings"].append(f"{name} trained on reduced sample due to MemoryError")
                except Exception as e2:
                    experiments.append({
                        "model_name": name, "status": "oom",
                        "primary_score": 0, "train_time_s": 0,
                        "primary_metric": primary_metric,
                    })
                    state["warnings"].append(f"{name} OOM even on reduced sample: {str(e2)[:80]}")

            except Exception as e:
                error_msg = str(e)[:200]
                experiments.append({
                    "model_name": name, "status": "error",
                    "error": error_msg, "primary_score": 0,
                    "primary_metric": primary_metric,
                })
                state["warnings"].append(f"{name} failed: {error_msg[:100]}")

            state["progress_pct"] = 42 + int((i + 1) / n_models * 28)

        completed = [e for e in experiments if e.get("status") in ("completed", "completed_reduced")]
        real_completed = [e for e in completed if not e.get("is_baseline")]
        if not real_completed:
            state["errors"].append("All models failed to train. Check the dataset and features.")
            state["status"] = "failed"
            return state

        best = max(real_completed, key=lambda e: e["primary_score"])
        baseline = next((e for e in completed if e.get("is_baseline")), None)
        state["experiments"] = experiments
        state["baseline_score"] = baseline
        if baseline:
            lift = best["primary_score"] - baseline["primary_score"]
            state["messages"].append(
                f"✓ Training complete. Best: {best['model_name']} "
                f"({best['primary_metric']}={best['primary_score']:.4f}, "
                f"baseline={baseline['primary_score']:.4f}, lift={lift:+.4f})"
            )
        else:
            state["messages"].append(
                f"✓ Training complete. Best: {best['model_name']} "
                f"({best['primary_metric']}={best['primary_score']:.4f}) — baseline failed to train"
            )
        state["progress_pct"] = 70

    except Exception as e:
        error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
        state["errors"].append(f"Experiment agent: {error_msg}")
        state["messages"].append(f"⚠ Training error: {error_msg[:100]}")

    return state
