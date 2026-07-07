import asyncio
import base64
import json
from io import BytesIO

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.llm_factory import get_llm
from app.tools.ml_tools import CLASSIFICATION_MODELS, REGRESSION_MODELS
from app.tools.stats_tools import bootstrap_metric_ci, compare_to_baseline

from .state import AgentState


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return b64


MODEL_EXPLANATION_FALLBACK = (
    "The model was successfully trained and evaluated. "
    "See the metrics above for detailed performance information."
)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm_model_card(llm, best_name, problem_type, target, description, metrics, top_features):
    prompt = f"""You are explaining an ML model to a business stakeholder (non-technical).

Model: {best_name}
Task: {problem_type} — predicting "{target}"
User goal: "{description}"
Metrics: {json.dumps(metrics, indent=2)}
Top features: {json.dumps(top_features, indent=2)}

Write a clear 150-200 word explanation covering:
1. What the model predicts and how accurate it is (use plain language, not jargon)
2. The 2-3 most important factors driving predictions
3. One honest limitation or caveat

Write in plain prose, no bullet points, no headers."""
    response = await asyncio.to_thread(llm.invoke, prompt)
    return response.content.strip()


async def run_evaluation_agent(state: AgentState) -> AgentState:
    state["current_agent"] = "evaluation"
    state["progress_pct"] = 72
    state["messages"].append("📊 Evaluating best model...")

    try:
        if not state.get("experiments"):
            state["errors"].append("No experiments found to evaluate.")
            state["status"] = "failed"
            return state

        if not state.get("preprocessor_path"):
            state["errors"].append("No preprocessor pipeline found.")
            state["status"] = "failed"
            return state

        llm = get_llm(state["llm_config"])

        # Load raw dataframe and unfitted preprocessor
        df = pd.read_parquet(state["processed_dataset_path"])
        target = state["target_column"]
        X = df.drop(columns=[target])
        y = df[target]

        preprocessor = joblib.load(state["preprocessor_path"])

        # Encode string target
        class_names = None
        if state["problem_type"] == "classification":
            if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
                class_names = le.classes_

        n_classes = y.nunique() if state["problem_type"] == "classification" else None
        is_binary = (state["problem_type"] == "classification" and n_classes == 2)
        primary_metric = (
            "roc_auc" if is_binary
            else "f1_weighted" if state["problem_type"] == "classification"
            else "r2"
        )

        # Select best model (baseline dummy is excluded — it exists only for comparison)
        completed = [e for e in state["experiments"]
                     if e.get("status") in ("completed", "completed_reduced")
                     and not e.get("is_baseline")]
        if not completed:
            state["errors"].append("No non-baseline experiments available to evaluate.")
            state["status"] = "failed"
            return state
        best_exp = max(completed, key=lambda e: e.get("primary_score", 0))
        best_name = best_exp["model_name"]
        state["best_model_name"] = best_name
        state["messages"].append(f"  Best model: {best_name}")

        # Train / test split
        stratify = y if state["problem_type"] == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        # Build and fit pipeline on train split
        models = (CLASSIFICATION_MODELS if state["problem_type"] == "classification"
                  else REGRESSION_MODELS)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", models[best_name]()),
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = None

        # ── METRICS ──────────────────────────────────────────────────────
        metrics: dict = {}

        if state["problem_type"] == "classification":
            from sklearn.metrics import accuracy_score, classification_report, f1_score
            metrics["test_accuracy"] = float(accuracy_score(y_test, y_pred))
            metrics["test_f1_weighted"] = float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            )
            metrics["classification_report"] = classification_report(
                y_test, y_pred, output_dict=True, zero_division=0
            )
            try:
                from sklearn.metrics import roc_auc_score
                y_prob = pipeline.predict_proba(X_test)
                if is_binary:
                    metrics["test_roc_auc"] = float(roc_auc_score(y_test, y_prob[:, 1]))
                else:
                    metrics["test_roc_auc"] = float(
                        roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
                    )
            except Exception:
                pass
        else:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            metrics["test_rmse"] = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics["test_mae"] = float(mean_absolute_error(y_test, y_pred))
            metrics["test_r2"] = float(r2_score(y_test, y_pred))
            y_test_arr = np.array(y_test)
            if not (y_test_arr == 0).any() and len(y_test_arr) > 0:
                metrics["test_mape"] = float(
                    np.mean(np.abs((y_test_arr - y_pred) / y_test_arr)) * 100
                )

        # Attach CV scores from experiment for comparison (Bug 6)
        cv_scores = best_exp.get("cv_scores", {})
        pm_cv_key = f"test_{primary_metric}"
        if pm_cv_key in cv_scores:
            metrics[f"cv_{primary_metric}_mean"] = cv_scores[pm_cv_key]["mean"]
            metrics[f"cv_{primary_metric}_std"] = cv_scores[pm_cv_key]["std"]

        state["evaluation_metrics"] = metrics

        # ── STATISTICAL RIGOR — kills "bare accuracy number" reporting ────
        try:
            y_true_arr = np.asarray(y_test)
            if state["problem_type"] == "classification":
                from sklearn.metrics import accuracy_score, roc_auc_score
                if is_binary and y_prob is not None:
                    ci_metric_fn = roc_auc_score
                    ci_pred_values = y_prob[:, 1]
                else:
                    ci_metric_fn = accuracy_score
                    ci_pred_values = np.asarray(y_pred)
            else:
                from sklearn.metrics import r2_score
                ci_metric_fn = r2_score
                ci_pred_values = np.asarray(y_pred)

            test_set_ci = bootstrap_metric_ci(y_true_arr, ci_pred_values, ci_metric_fn)

            baseline = state.get("baseline_score")
            vs_baseline = None
            if baseline:
                vs_baseline = compare_to_baseline(
                    best_exp.get("primary_score_folds", []),
                    baseline.get("primary_score_folds", []),
                )

            state["statistical_comparison"] = {
                "primary_metric": primary_metric,
                "test_set_ci": test_set_ci,
                "vs_baseline": vs_baseline,
            }
            if vs_baseline and vs_baseline.get("probability_better_than_baseline") is not None:
                state["messages"].append(
                    f"  Statistical check: {vs_baseline['confidence']} confidence the model beats "
                    f"a naive baseline (P={vs_baseline['probability_better_than_baseline']:.2f}, "
                    f"lift={vs_baseline['lift']:+.4f})"
                )
        except Exception as e:
            state["warnings"].append(f"Statistical comparison failed: {str(e)[:100]}")
            state["statistical_comparison"] = None

        # Save full pipeline (preprocessor + model)
        import os
        os.makedirs("./data/storage", exist_ok=True)
        model_path = f"./data/storage/{state['job_id']}_best_model.joblib"
        joblib.dump(pipeline, model_path)
        state["best_model_path"] = model_path

        # ── EVALUATION CHARTS ─────────────────────────────────────────────
        charts = []

        # Feature importance — pull from the fitted model inside the pipeline
        fitted_model = pipeline.named_steps["model"]
        fitted_prep = pipeline.named_steps["preprocessor"]

        importances = None
        if hasattr(fitted_model, "feature_importances_"):
            importances = fitted_model.feature_importances_
        elif hasattr(fitted_model, "coef_"):
            coef = fitted_model.coef_
            importances = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)

        feat_imp = None
        top_features = {}
        if importances is not None:
            try:
                feat_names = list(fitted_prep.get_feature_names_out())
            except Exception:
                feat_names = [f"feature_{i}" for i in range(len(importances))]

            if len(importances) == len(feat_names):
                feat_imp = pd.Series(importances, index=feat_names).nlargest(20)
                top_features = feat_imp.head(5).to_dict()
                try:
                    fig, ax = plt.subplots(figsize=(10, 7))
                    feat_imp.sort_values().plot(kind="barh", ax=ax, color="steelblue")
                    ax.set_title(f"Top Feature Importances — {best_name}")
                    ax.set_xlabel("Importance")
                    plt.tight_layout()
                    charts.append({
                        "title": "Feature Importances", "chart_type": "bar",
                        "base64_png": _fig_to_b64(fig),
                        "description": f"Top {len(feat_imp)} features by importance for {best_name}",
                    })
                except Exception as e:
                    state["warnings"].append(f"Feature importance chart failed: {str(e)[:60]}")

        # ── SHAP EXPLAINABILITY — real per-feature attribution, not just a generic
        # feature_importances_ bar chart ──────────────────────────────────
        try:
            from app.tools.shap_tools import compute_shap_summary
            shap_summary = compute_shap_summary(pipeline, X_test, str(state["problem_type"]))
            state["shap_summary"] = shap_summary
            charts.append(shap_summary["bar_chart"])
            charts.append(shap_summary["waterfall_chart"])
            state["messages"].append("  SHAP explainability computed")
        except Exception as e:
            state["warnings"].append(f"SHAP computation failed: {str(e)[:100]}")
            state["shap_summary"] = None

        if state["problem_type"] == "classification":
            # Confusion matrix
            try:
                from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
                cm = confusion_matrix(y_test, y_pred, normalize="true")
                disp_labels = class_names if class_names is not None else sorted(set(y_test))
                if len(disp_labels) > 20:
                    disp_labels = None
                fig, ax = plt.subplots(figsize=(8, 6))
                ConfusionMatrixDisplay(cm, display_labels=disp_labels).plot(ax=ax, colorbar=True)
                ax.set_title("Confusion Matrix (normalized)")
                plt.tight_layout()
                charts.append({
                    "title": "Confusion Matrix", "chart_type": "confusion_matrix",
                    "base64_png": _fig_to_b64(fig),
                    "description": "Normalized confusion matrix on held-out test set",
                })
            except Exception as e:
                state["warnings"].append(f"Confusion matrix failed: {str(e)[:60]}")

            # ROC curve (binary only)
            if is_binary and "test_roc_auc" in metrics:
                try:
                    from sklearn.metrics import RocCurveDisplay
                    fig, ax = plt.subplots(figsize=(8, 6))
                    RocCurveDisplay.from_predictions(
                        y_test, pipeline.predict_proba(X_test)[:, 1], ax=ax
                    )
                    ax.set_title(f"ROC Curve — AUC={metrics['test_roc_auc']:.3f}")
                    plt.tight_layout()
                    charts.append({
                        "title": "ROC Curve", "chart_type": "roc",
                        "base64_png": _fig_to_b64(fig),
                        "description": f"ROC curve (test set) — AUC = {metrics['test_roc_auc']:.3f}",
                    })
                except Exception as e:
                    state["warnings"].append(f"ROC curve failed: {str(e)[:60]}")

        else:
            # Predicted vs Actual
            try:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(y_test, y_pred, alpha=0.4, color="steelblue", s=15)
                mn = min(float(np.min(y_test)), float(np.min(y_pred)))
                mx = max(float(np.max(y_test)), float(np.max(y_pred)))
                ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Perfect prediction")
                ax.set_xlabel("Actual")
                ax.set_ylabel("Predicted")
                ax.set_title(f"Predicted vs Actual — R²={metrics.get('test_r2', 0):.3f}")
                ax.legend()
                plt.tight_layout()
                charts.append({
                    "title": "Predicted vs Actual", "chart_type": "scatter",
                    "base64_png": _fig_to_b64(fig),
                    "description": f"Predicted vs actual values (test set). R² = {metrics.get('test_r2', 0):.3f}",
                })
            except Exception as e:
                state["warnings"].append(f"Predicted vs Actual chart failed: {str(e)[:60]}")

            # Residuals
            try:
                residuals = np.array(y_test) - y_pred
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(residuals, bins=40, color="steelblue", edgecolor="white")
                ax.axvline(0, color="red", linestyle="--")
                ax.set_xlabel("Residual")
                ax.set_title("Residuals Distribution")
                plt.tight_layout()
                charts.append({
                    "title": "Residuals Distribution", "chart_type": "histogram",
                    "base64_png": _fig_to_b64(fig),
                    "description": "Distribution of prediction errors on test set",
                })
            except Exception as e:
                state["warnings"].append(f"Residuals chart failed: {str(e)[:60]}")

        state["evaluation_charts"] = charts

        # LLM model card
        metrics_for_llm = {k: v for k, v in metrics.items() if k != "classification_report"}
        try:
            state["model_explanation"] = await _call_llm_model_card(
                llm, best_name, state["problem_type"],
                target, state["user_description"], metrics_for_llm, top_features
            )
        except Exception as e:
            error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
            state["warnings"].append(f"Model explanation LLM call failed: {error_msg[:100]}")
            state["model_explanation"] = MODEL_EXPLANATION_FALLBACK

        state["messages"].append(f"✓ Evaluation complete — best model: {best_name}")
        state["progress_pct"] = 90

    except Exception as e:
        error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
        state["errors"].append(f"Evaluation agent: {error_msg}")
        state["messages"].append(f"⚠ Evaluation error: {error_msg[:100]}")

    return state
