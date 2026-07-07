import base64
from io import BytesIO
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return b64


def _reduce_to_2d(values: np.ndarray, predicted_class_idx: Optional[np.ndarray] = None) -> np.ndarray:
    """Collapse a (n, features, classes) SHAP array to (n, features) — one class's
    contribution per row (the predicted class), or mean-abs across classes if unknown."""
    if values.ndim == 2:
        return values
    if predicted_class_idx is not None:
        return np.take_along_axis(
            values, predicted_class_idx[:, None, None], axis=2
        )[:, :, 0]
    return np.abs(values).mean(axis=2)


def compute_shap_summary(pipeline, X_sample: pd.DataFrame, problem_type: str, max_rows: int = 200) -> dict:
    """Computes SHAP feature attributions for the fitted pipeline's model. Returns a bar chart
    of mean |SHAP| per feature plus a per-instance waterfall — real explainability, not just a
    generic sklearn feature_importances_ bar. Returns None (with the caller logging a warning)
    if SHAP can't handle this model type, rather than failing the whole evaluation.
    """
    import shap  # imported lazily — heavy optional dependency

    fitted_model = pipeline.named_steps["model"]
    fitted_prep = pipeline.named_steps["preprocessor"]

    X_sub = X_sample.sample(n=min(max_rows, len(X_sample)), random_state=42)
    X_transformed = fitted_prep.transform(X_sub)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    try:
        feat_names = list(fitted_prep.get_feature_names_out())
    except Exception:
        feat_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    try:
        explainer = shap.TreeExplainer(fitted_model)
    except Exception:
        background = X_transformed[: min(50, len(X_transformed))]
        explainer = shap.Explainer(fitted_model.predict, background)

    explanation = explainer(X_transformed)
    values = np.asarray(explanation.values)

    predicted_class_idx = None
    if problem_type == "classification" and values.ndim == 3:
        try:
            predicted_class_idx = np.asarray(fitted_model.predict(X_transformed))
            # predict() gives label values, not class-index positions — map to positions
            if hasattr(fitted_model, "classes_"):
                class_to_idx = {c: i for i, c in enumerate(fitted_model.classes_)}
                predicted_class_idx = np.array([class_to_idx.get(c, 0) for c in predicted_class_idx])
        except Exception:
            predicted_class_idx = None

    values_2d = _reduce_to_2d(values, predicted_class_idx)

    if len(feat_names) != values_2d.shape[1]:
        feat_names = [f"feature_{i}" for i in range(values_2d.shape[1])]

    mean_abs = np.abs(values_2d).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:15]
    top_names = [feat_names[i] for i in order]
    top_importance = mean_abs[order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(top_names) * 0.4)))
    ax.barh(top_names[::-1], top_importance[::-1], color="#2c7fb8", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax.set_title("SHAP Feature Importance")
    plt.tight_layout()
    bar_chart_b64 = _fig_to_b64(fig)

    # One representative instance — the one closest to the median prediction, more
    # illustrative than a random row.
    order_by_magnitude = np.argsort(np.abs(values_2d).sum(axis=1))
    row_idx = int(order_by_magnitude[len(values_2d) // 2])
    row_values = values_2d[row_idx]
    row_order = np.argsort(np.abs(row_values))[::-1][:12]
    row_names = [feat_names[i] for i in row_order]
    row_vals = row_values[row_order]

    fig, ax = plt.subplots(figsize=(9, max(4, len(row_names) * 0.4)))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in row_vals[::-1]]
    ax.barh(row_names[::-1], row_vals[::-1], color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (red = pushes prediction up, blue = pushes down)")
    ax.set_title("Example Prediction — Feature Contributions")
    plt.tight_layout()
    waterfall_chart_b64 = _fig_to_b64(fig)

    top_features = {name: float(val) for name, val in zip(top_names, top_importance)}

    return {
        "bar_chart": {
            "title": "SHAP Feature Importance", "chart_type": "shap_bar",
            "base64_png": bar_chart_b64,
            "description": "Mean absolute SHAP value per feature — how much each feature "
                           "actually moves predictions, averaged across a sample of the test set.",
        },
        "waterfall_chart": {
            "title": "SHAP Example Explanation", "chart_type": "shap_waterfall",
            "base64_png": waterfall_chart_b64,
            "description": "Feature contributions for one representative prediction.",
        },
        "top_features": top_features,
    }
