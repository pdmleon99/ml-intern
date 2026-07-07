import base64
from io import BytesIO
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return b64


def chart_target_distribution(df: pd.DataFrame, target: str, problem_type: str) -> dict:
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use("seaborn-v0_8-whitegrid")
    y = df[target].dropna()

    if problem_type == "classification":
        vc = y.value_counts()
        vc.plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(f"Target Distribution — {target}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        desc = f"Class distribution for target '{target}'"
    else:
        y.hist(ax=ax, bins=40, color="steelblue", edgecolor="white")
        try:
            y_clean = y.astype(float).dropna()
            if len(y_clean) > 1:
                from scipy.stats import gaussian_kde
                ax2 = ax.twinx()
                kde = gaussian_kde(y_clean)
                xs = np.linspace(y_clean.min(), y_clean.max(), 200)
                ax2.plot(xs, kde(xs), "r-", linewidth=2)
                ax2.set_ylabel("Density")
        except Exception:
            pass
        ax.set_title(f"Target Distribution — {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        desc = f"Distribution of target variable '{target}'"

    b64 = _fig_to_b64(fig)
    return {"title": "Target Distribution", "chart_type": "distribution",
            "base64_png": b64, "description": desc}


def _text_summary_chart(line1: str, line2: str = "") -> str:
    """Return a base64 PNG containing just two lines of centered text."""
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    ax.text(0.5, 0.6, line1, ha="center", va="center",
            fontsize=14, color="#27ae60", fontweight="bold", transform=ax.transAxes)
    if line2:
        ax.text(0.5, 0.35, line2, ha="center", va="center",
                fontsize=11, color="#555555", transform=ax.transAxes)
    plt.tight_layout()
    return _fig_to_b64(fig)


def chart_missing_values(df: pd.DataFrame, overall_missing_pct: float = None) -> dict:
    """Always returns a chart — either a bar chart or a text-summary placeholder."""
    if overall_missing_pct is None:
        overall_missing_pct = float(df.isnull().mean().mean())

    cols_with_missing = [c for c in df.columns if df[c].isnull().any()]

    # Only show heatmap when missing data is meaningful
    if overall_missing_pct > 0.01 and len(cols_with_missing) >= 2:
        missing = df[cols_with_missing].isnull().mean().sort_values(ascending=False)
        top = missing.head(30)
        fig, ax = plt.subplots(figsize=(10, max(4, len(top) * 0.35)))
        top.sort_values().plot(kind="barh", ax=ax, color="coral", edgecolor="white")
        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Missing Fraction")
        ax.axvline(0.1, color="orange", linestyle="--", alpha=0.7, label="10%")
        ax.axvline(0.4, color="red", linestyle="--", alpha=0.7, label="40%")
        ax.legend()
        b64 = _fig_to_b64(fig)
        return {"title": "Missing Values", "chart_type": "missing_heatmap",
                "base64_png": b64,
                "description": f"{len(cols_with_missing)} columns have missing values"}
    else:
        b64 = _text_summary_chart(
            "✓ No significant missing values detected.",
            f"Only {overall_missing_pct:.2%} of values are missing across all columns."
        )
        return {"title": "Missing Values", "chart_type": "text_summary",
                "base64_png": b64,
                "description": "Dataset has no significant missing value issues."}


def chart_correlation_heatmap(df: pd.DataFrame, target: str) -> Optional[dict]:
    """Feature-to-feature correlation heatmap — target column excluded."""
    num_df = df.select_dtypes(include="number")
    # Exclude target from both rows and columns
    feature_cols = [c for c in num_df.columns if c != target]
    if len(feature_cols) < 2:
        return None

    # Limit to top-20 by variance
    sub = df[feature_cols]
    if len(feature_cols) > 20:
        top_var = sub.var().nlargest(20).index.tolist()
        sub = sub[top_var]

    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=len(corr) <= 12, fmt=".2f",
                cmap="coolwarm", center=0, ax=ax,
                linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature-to-Feature Correlations (target excluded)")
    b64 = _fig_to_b64(fig)
    return {"title": "Correlation Heatmap", "chart_type": "heatmap",
            "base64_png": b64,
            "description": "Pairwise correlations among numeric features (target not included)"}


def chart_target_correlation_bar(df: pd.DataFrame, target: str) -> Optional[dict]:
    """Horizontal bar chart of |correlation| between each feature and the target."""
    if target not in df.columns:
        return None
    if not pd.api.types.is_numeric_dtype(df[target]):
        return None

    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target]
    if not feature_cols:
        return None

    corr = df[feature_cols].corrwith(df[target]).abs().dropna().sort_values(ascending=True)
    if corr.empty:
        return None

    # Limit to top-20
    corr = corr.tail(20)
    fig, ax = plt.subplots(figsize=(10, max(4, len(corr) * 0.4)))
    colors = ["#e74c3c" if v > 0.5 else "#3498db" for v in corr.values]
    corr.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Feature Correlation with Target: {target}")
    ax.set_xlabel("|Pearson r|")
    ax.axvline(0.1, color="gray", linestyle="--", alpha=0.5, label="r=0.1")
    ax.axvline(0.5, color="orange", linestyle="--", alpha=0.5, label="r=0.5")
    ax.legend(fontsize=8)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    return {"title": "Feature–Target Correlations", "chart_type": "bar",
            "base64_png": b64,
            "description": f"Absolute Pearson correlation of each feature with '{target}' (red = strong, blue = weak)"}


def chart_top_features_vs_target(df: pd.DataFrame, target: str, problem_type: str) -> Optional[dict]:
    num_df = df.select_dtypes(include="number")
    if target not in num_df.columns and problem_type == "regression":
        return None

    if problem_type == "regression" and target in num_df.columns:
        corr = num_df.corr()[target].abs().drop(target, errors="ignore")
        top_features = corr.nlargest(5).index.tolist()
        if not top_features:
            return None
        n = len(top_features)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, feat in zip(axes, top_features):
            ax.scatter(df[feat], df[target], alpha=0.3, s=8, color="steelblue")
            ax.set_xlabel(feat[:20])
            ax.set_ylabel(target[:20])
            ax.set_title(f"r={df[feat].corr(df[target]):.2f}")
        plt.suptitle("Top Features vs Target", y=1.02)
        plt.tight_layout()
        b64 = _fig_to_b64(fig)
        return {"title": "Top Features vs Target", "chart_type": "scatter",
                "base64_png": b64,
                "description": "Scatter plots of top correlated features vs target"}

    elif problem_type == "classification":
        num_cols = [c for c in num_df.columns if c != target][:5]
        if not num_cols:
            return None
        n = len(num_cols)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
        if n == 1:
            axes = [axes]
        for ax, feat in zip(axes, num_cols):
            df.boxplot(column=feat, by=target, ax=ax)
            ax.set_title(feat[:20])
            ax.set_xlabel("")
        plt.suptitle("Numeric Features by Class")
        plt.tight_layout()
        b64 = _fig_to_b64(fig)
        return {"title": "Features by Class", "chart_type": "boxplot",
                "base64_png": b64,
                "description": "Distribution of numeric features per class"}

    return None


def chart_class_balance(df: pd.DataFrame, target: str) -> Optional[dict]:
    vc = df[target].value_counts()
    if len(vc) < 2 or len(vc) > 30:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(vc)))
    bars = ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="white")
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + vc.values.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9)
    ax.set_title(f"Class Balance — {target}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    b64 = _fig_to_b64(fig)
    pcts = (vc / vc.sum() * 100).round(1).to_dict()
    return {"title": "Class Balance", "chart_type": "bar",
            "base64_png": b64, "description": f"Class distribution: {pcts}"}


def chart_categorical_top_values(df: pd.DataFrame, target: str) -> Optional[dict]:
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != target][:10]
    if not cat_cols:
        return None
    n = min(len(cat_cols), 10)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 3.5))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cat_cols):
        ax = axes[i]
        vc = df[col].value_counts().head(8)
        vc.sort_values().plot(kind="barh", ax=ax, color="teal", edgecolor="white")
        ax.set_title(col[:30])
        ax.set_xlabel("Count")
        ax.tick_params(axis="y", labelsize=8)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Top Categorical Values", y=1.01)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    return {"title": "Categorical Value Counts", "chart_type": "bar_grid",
            "base64_png": b64, "description": f"Top values for {n} categorical columns"}
