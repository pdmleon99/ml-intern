import base64
from io import BytesIO
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from app.tools.chart_style import DIVERGING_CMAP, PALETTE, SEQUENCE, apply_style, style_ax

apply_style()


def _fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    return b64


def chart_target_distribution(df: pd.DataFrame, target: str, problem_type: str) -> dict:
    fig, ax = plt.subplots(figsize=(10, 5))
    y = df[target].dropna()

    if problem_type == "classification":
        vc = y.value_counts()
        vc.plot(kind="bar", ax=ax, color=PALETTE["primary"], edgecolor="white", width=0.7)
        ax.set_title(f"Target Distribution — {target}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=45)
        desc = f"Class distribution for target '{target}'"
    else:
        y.hist(ax=ax, bins=40, color=PALETTE["primary"], edgecolor="white", alpha=0.85, grid=False)
        try:
            y_clean = y.astype(float).dropna()
            if len(y_clean) > 1:
                from scipy.stats import gaussian_kde
                ax2 = ax.twinx()
                kde = gaussian_kde(y_clean)
                xs = np.linspace(y_clean.min(), y_clean.max(), 200)
                ax2.plot(xs, kde(xs), color=PALETTE["danger"], linewidth=2.2)
                ax2.set_ylabel("Density")
                ax2.grid(False)
                style_ax(ax2)
        except Exception:
            pass
        ax.set_title(f"Target Distribution — {target}")
        ax.set_xlabel(target)
        ax.set_ylabel("Count")
        desc = f"Distribution of target variable '{target}'"

    style_ax(ax)
    b64 = _fig_to_b64(fig)
    return {"title": "Target Distribution", "chart_type": "distribution",
            "base64_png": b64, "description": desc}


def _text_summary_chart(line1: str, line2: str = "") -> str:
    """Return a base64 PNG containing just two lines of centered text."""
    from matplotlib.patches import Ellipse

    figsize = (8, 3)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # A drawn checkmark badge instead of the ✓ unicode glyph — that glyph is missing from
    # several common fonts (renders as a broken tofu box), a vector shape always renders.
    # Ellipse width/height are compensated for the figure's aspect ratio so it reads as a
    # circle rather than a squashed oval (axes-fraction coords aren't 1:1 in pixels here).
    aspect = figsize[1] / figsize[0]
    badge_x, badge_y, badge_r = 0.5 - len(line1) * 0.0068 - 0.05, 0.62, 0.05
    ax.add_patch(Ellipse(
        (badge_x, badge_y), width=badge_r * aspect, height=badge_r,
        color=PALETTE["success"], transform=ax.transAxes, zorder=2,
    ))
    ax.plot(
        [badge_x - badge_r * aspect * 0.28, badge_x - badge_r * aspect * 0.05, badge_x + badge_r * aspect * 0.3],
        [badge_y - badge_r * 0.05, badge_y - badge_r * 0.22, badge_y + badge_r * 0.25],
        color="white", linewidth=2.2, solid_capstyle="round", solid_joinstyle="round",
        transform=ax.transAxes, zorder=3,
    )
    ax.text(0.5 + 0.02, 0.6, line1, ha="center", va="center",
            fontsize=15, color=PALETTE["success"], fontweight="bold", transform=ax.transAxes)
    if line2:
        ax.text(0.5, 0.35, line2, ha="center", va="center",
                fontsize=11, color=PALETTE["text_muted"], transform=ax.transAxes)
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
        top.sort_values().plot(kind="barh", ax=ax, color=PALETTE["warning"], edgecolor="white")
        ax.set_title("Missing Values by Column")
        ax.set_xlabel("Missing Fraction")
        ax.axvline(0.1, color=PALETTE["neutral"], linestyle="--", alpha=0.6, label="10%")
        ax.axvline(0.4, color=PALETTE["danger"], linestyle="--", alpha=0.6, label="40%")
        ax.legend()
        style_ax(ax)
        b64 = _fig_to_b64(fig)
        return {"title": "Missing Values", "chart_type": "missing_heatmap",
                "base64_png": b64,
                "description": f"{len(cols_with_missing)} columns have missing values"}
    else:
        b64 = _text_summary_chart(
            "No significant missing values detected.",
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
                cmap=DIVERGING_CMAP, center=0, ax=ax,
                linewidths=0.5, linecolor="white", square=True, cbar_kws={"shrink": 0.8})
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
    colors = [PALETTE["danger"] if v > 0.5 else PALETTE["accent"] for v in corr.values]
    corr.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
    ax.set_title(f"Feature Correlation with Target: {target}")
    ax.set_xlabel("|Pearson r|")
    ax.axvline(0.1, color=PALETTE["neutral"], linestyle="--", alpha=0.5, label="r=0.1")
    ax.axvline(0.5, color=PALETTE["warning"], linestyle="--", alpha=0.5, label="r=0.5")
    ax.legend(fontsize=8)
    style_ax(ax)
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
            ax.scatter(df[feat], df[target], alpha=0.35, s=10, color=PALETTE["primary"], edgecolor="none")
            ax.set_xlabel(feat[:20])
            ax.set_ylabel(target[:20])
            ax.set_title(f"r={df[feat].corr(df[target]):.2f}")
            style_ax(ax)
        fig.suptitle("Top Features vs Target", fontweight="bold", color=PALETTE["text"], fontsize=13)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
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
            bp = df.boxplot(column=feat, by=target, ax=ax, patch_artist=True, return_type="dict")
            for box in bp[feat]["boxes"]:
                box.set_facecolor(PALETTE["primary_light"])
                box.set_edgecolor(PALETTE["primary"])
            for median in bp[feat]["medians"]:
                median.set_color(PALETTE["secondary"])
            ax.set_title(feat[:20])
            ax.set_xlabel("")
            style_ax(ax)
        # pandas' boxplot(by=...) auto-adds its own suptitle Text artist — rewriting it in
        # place (rather than clearing + fig.suptitle()) is what actually renders reliably;
        # fig.suptitle() after a pandas boxplot silently no-ops in this matplotlib version.
        for t in fig.texts:
            t.set_text("Numeric Features by Class")
            t.set_fontweight("bold")
            t.set_color(PALETTE["text"])
            t.set_fontsize(13)
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
    colors = [SEQUENCE[i % len(SEQUENCE)] for i in range(len(vc))]
    bars = ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="white", width=0.65)
    for bar, val in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + vc.values.max() * 0.01,
                f"{val:,}", ha="center", va="bottom", fontsize=9, color=PALETTE["text"])
    ax.set_title(f"Class Balance — {target}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    style_ax(ax)
    b64 = _fig_to_b64(fig)
    pcts = (vc / vc.sum() * 100).round(1).to_dict()
    return {"title": "Class Balance", "chart_type": "bar",
            "base64_png": b64, "description": f"Class distribution: {pcts}"}


def chart_categorical_top_values(df: pd.DataFrame, target: str) -> Optional[dict]:
    cat_cols = [c for c in df.select_dtypes(include="object").columns if c != target][:10]
    if not cat_cols:
        return None
    n = min(len(cat_cols), 10)
    ncols = min(2, n)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, nrows * 3.5))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cat_cols):
        ax = axes[i]
        vc = df[col].value_counts().head(8)
        vc.sort_values().plot(kind="barh", ax=ax, color=PALETTE["accent"], edgecolor="white")
        ax.set_title(col[:30])
        ax.set_xlabel("Count")
        ax.tick_params(axis="y", labelsize=8)
        style_ax(ax)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Top Categorical Values", fontweight="bold", color=PALETTE["text"], fontsize=13)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    b64 = _fig_to_b64(fig)
    return {"title": "Categorical Value Counts", "chart_type": "bar_grid",
            "base64_png": b64, "description": f"Top values for {n} categorical columns"}
