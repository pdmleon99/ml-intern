import asyncio
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings
from app.core.llm_factory import get_llm
from app.tools.dataset_tools import (
    build_dataset_profile,
    load_dataframe,
    save_sample_parquet,
)
from app.tools.viz_tools import (
    chart_categorical_top_values,
    chart_class_balance,
    chart_correlation_heatmap,
    chart_missing_values,
    chart_target_correlation_bar,
    chart_target_distribution,
    chart_top_features_vs_target,
)

from .state import AgentState

EDA_NARRATIVE_FALLBACK = {
    "summary": "Dataset loaded and profiled successfully. See findings below for details.",
    "key_findings": ["Profile computed", "Findings generated", "Charts created"],
    "fe_recommendations": ["Apply imputation", "Encode categoricals", "Scale numeric features"],
}


def _detect_target_from_description(description: str, columns: list) -> Optional[str]:
    desc_lower = description.lower()
    for col in columns:
        if col.lower() in desc_lower:
            return col
    # fuzzy: check substrings
    for col in columns:
        parts = re.split(r"[_\s-]", col.lower())
        if any(p in desc_lower for p in parts if len(p) > 3):
            return col
    return None


def _detect_problem_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() > 20:
            return "regression"
    return "classification"


def _generate_findings(df: pd.DataFrame, col_profiles: dict, target: str, problem_type: str) -> list:
    findings = []

    for col, profile in col_profiles.items():
        mp = profile.get("missing_pct", 0)
        if mp > 0.4 and col != target:
            findings.append({"finding": f"{mp:.0%} of values are missing", "severity": "critical", "column": col})
        elif 0.1 < mp <= 0.4:
            findings.append({"finding": f"{mp:.0%} of values are missing", "severity": "warning", "column": col})

        if profile.get("is_constant"):
            findings.append({"finding": "Column is constant (zero variance)", "severity": "critical", "column": col})

        if profile.get("near_zero_variance"):
            findings.append({"finding": "Near-zero variance", "severity": "critical", "column": col})

        if profile.get("is_high_cardinality") and not profile.get("is_id"):
            findings.append({"finding": f"High cardinality ({profile['n_unique']} unique values)", "severity": "warning", "column": col})

        if profile.get("is_id"):
            findings.append({"finding": "Likely an ID column (high uniqueness, no predictive value)", "severity": "warning", "column": col})

    # Target-specific
    if target in df.columns:
        y = df[target].dropna()
        if problem_type == "classification":
            vc = y.value_counts(normalize=True)
            min_pct = vc.min()
            if min_pct < 0.05:
                findings.append({"finding": f"Severe class imbalance (minority={min_pct:.1%})", "severity": "critical", "column": target})
            elif min_pct < 0.1:
                findings.append({"finding": f"Class imbalance (minority={min_pct:.1%})", "severity": "warning", "column": target})

        # Leakage detection
        num_df = df.select_dtypes(include="number")
        if target in num_df.columns:
            for col in num_df.columns:
                if col == target:
                    continue
                try:
                    corr = abs(num_df[col].corr(num_df[target]))
                    if corr > 0.98:
                        findings.append({"finding": f"Extremely high correlation with target ({corr:.3f}) — possible data leakage", "severity": "critical", "column": col})
                except Exception:
                    pass

    # Dataset-level
    total_rows = len(df)
    dup_pct = df.duplicated().mean()
    if dup_pct > 0.1:
        findings.append({"finding": f"{dup_pct:.1%} duplicate rows", "severity": "warning", "column": "dataset"})

    if total_rows < 500:
        findings.append({"finding": f"Very small dataset ({total_rows} rows) — model reliability may be limited", "severity": "warning", "column": "dataset"})

    overall_missing = df.isnull().mean().mean()
    if overall_missing > 0.3:
        findings.append({"finding": f"High overall missing rate ({overall_missing:.1%})", "severity": "warning", "column": "dataset"})

    # Single-class check
    if target in df.columns and df[target].nunique() <= 1:
        findings.append({"finding": "Target column has only one unique value — cannot train", "severity": "critical", "column": target})

    return findings


def _generate_all_charts(df_work: pd.DataFrame, target: str, problem_type: str, profile: dict) -> tuple[list, list]:
    """All 7 matplotlib charts, run synchronously off the event loop via asyncio.to_thread —
    rendering charts one at a time on the event loop thread was blocking SSE delivery to the
    frontend for the whole EDA stage (visible as the UI looking frozen during live testing)."""
    charts: list = []
    warnings: list = []

    try:
        c = chart_target_distribution(df_work, target, problem_type)
        if c:
            charts.append(c)
    except Exception as e:
        warnings.append(f"Target distribution chart failed: {str(e)[:80]}")

    try:
        overall_mp = profile.get("missing_pct", 0.0)
        c = chart_missing_values(df_work, overall_missing_pct=overall_mp)
        charts.append(c)
    except Exception as e:
        warnings.append(f"Missing values chart failed: {str(e)[:80]}")

    try:
        c = chart_correlation_heatmap(df_work, target)
        if c:
            charts.append(c)
    except Exception as e:
        warnings.append(f"Correlation heatmap failed: {str(e)[:80]}")

    try:
        c = chart_target_correlation_bar(df_work, target)
        if c:
            charts.append(c)
    except Exception as e:
        warnings.append(f"Target correlation bar failed: {str(e)[:80]}")

    try:
        c = chart_top_features_vs_target(df_work, target, problem_type)
        if c:
            charts.append(c)
    except Exception as e:
        warnings.append(f"Features vs target chart failed: {str(e)[:80]}")

    if problem_type == "classification":
        try:
            c = chart_class_balance(df_work, target)
            if c:
                charts.append(c)
        except Exception as e:
            warnings.append(f"Class balance chart failed: {str(e)[:80]}")

    try:
        c = chart_categorical_top_values(df_work, target)
        if c:
            charts.append(c)
    except Exception as e:
        warnings.append(f"Categorical chart failed: {str(e)[:80]}")

    return charts, warnings


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm_for_narrative(llm, compact_profile: dict) -> dict:
    prompt = f"""You are a data scientist reviewing a dataset.
Profile: {json.dumps(compact_profile, indent=2)}

Write a JSON response with exactly these three fields:
{{
  "summary": "3 sentences describing dataset quality and key characteristics",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "fe_recommendations": ["recommendation 1", "recommendation 2", "recommendation 3"]
}}
Reply with ONLY the JSON, no markdown, no explanation."""
    response = (await asyncio.to_thread(llm.invoke, prompt)).content
    response = response.replace("```json", "").replace("```", "").strip()
    return json.loads(response)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def _call_llm_for_target(llm, columns: list, description: str) -> str:
    prompt = f"""Dataset columns: {columns}
User goal: "{description}"
Which column is most likely the prediction target?
Reply with ONLY the column name, nothing else."""
    response = await asyncio.to_thread(llm.invoke, prompt)
    return response.content.strip().strip('"').strip("'")


async def run_eda_agent(state: AgentState) -> AgentState:
    state["current_agent"] = "eda"
    state["progress_pct"] = 5
    state["messages"].append("🔍 Loading and profiling dataset...")

    try:
        llm = get_llm(state["llm_config"])

        # ── 1. LOAD DATASET ──────────────────────────────────────────────
        file_path = state["dataset_path"]
        df = await asyncio.to_thread(load_dataframe, file_path)

        # Guard: empty dataset
        if df.empty or len(df.columns) == 0:
            state["errors"].append("Dataset is empty or has no columns.")
            state["status"] = "failed"
            return state

        # Guard: all text / no numeric/categorical structure
        all_obj = all(pd.api.types.is_object_dtype(df[c]) for c in df.columns)
        if all_obj and all(df[c].apply(lambda x: len(str(x).split()) > 5 if pd.notna(x) else False).mean() > 0.5 for c in df.columns):
            state["errors"].append("Dataset appears to be all free text — only tabular datasets are supported.")
            state["status"] = "failed"
            return state

        # Handle wide datasets: limit to top 200 cols by variance
        if len(df.columns) > settings.MAX_COLS_FULL_ANALYSIS:
            state["warnings"].append(f"Dataset has {len(df.columns)} columns — limiting to top {settings.MAX_COLS_FULL_ANALYSIS} by variance.")
            num_df = df.select_dtypes(include="number")
            if len(num_df.columns) > 0:
                top_var = num_df.var().nlargest(settings.MAX_COLS_FULL_ANALYSIS).index.tolist()
                other_cols = [c for c in df.columns if c not in num_df.columns]
                df = df[other_cols + top_var]

        original_n_rows = len(df)
        state["original_columns"] = list(df.columns)

        # Sampling for large datasets
        is_large = original_n_rows > settings.LARGE_DATASET_THRESHOLD
        if is_large:
            sample_n = min(settings.SAMPLE_SIZE_LARGE, original_n_rows)
            df_work = df.sample(n=sample_n, random_state=42)
            sample_path = save_sample_parquet(df_work, file_path)
            state["df_sample_path"] = sample_path
            state["messages"].append(f"  Dataset has {original_n_rows:,} rows — using {sample_n:,} row sample for analysis")
        else:
            df_work = df
            sample_path = file_path.replace(Path(file_path).suffix, "_sample.parquet")
            df_work.to_parquet(sample_path, index=False)
            state["df_sample_path"] = sample_path

        state["messages"].append(f"  Loaded: {original_n_rows:,} rows × {len(df.columns)} columns")
        state["progress_pct"] = 8

        # ── 2. COLUMN TYPE INFERENCE + PROFILE ───────────────────────────
        profile = await asyncio.to_thread(build_dataset_profile, df_work, original_n_rows, is_large)
        state["profile"] = profile

        # ── 3. TARGET COLUMN DETECTION ───────────────────────────────────
        if not state.get("target_column"):
            # Try regex/fuzzy from description
            guessed = _detect_target_from_description(
                state["user_description"], list(df.columns)
            )
            if guessed and guessed in df.columns:
                state["target_column"] = guessed
                state["messages"].append(f"  Auto-detected target from description: '{guessed}'")
            else:
                # Ask LLM
                try:
                    llm_target = await _call_llm_for_target(llm, list(df.columns), state["user_description"])
                    if llm_target in df.columns:
                        state["target_column"] = llm_target
                    else:
                        # Fallback: last column heuristic
                        state["target_column"] = df.columns[-1]
                        state["warnings"].append(f"Could not reliably detect target — defaulting to last column: '{df.columns[-1]}'")
                except Exception:
                    state["target_column"] = df.columns[-1]
                    state["warnings"].append(f"LLM target detection failed — defaulting to: '{df.columns[-1]}'")

        target = state["target_column"]

        # Validate target exists
        if not target or target not in df.columns:
            state["errors"].append(f"Target column '{target}' not found in dataset. Columns: {list(df.columns)[:10]}")
            state["status"] = "failed"
            return state

        # Detect problem type
        if not state.get("problem_type") or state["problem_type"] == "unknown":
            state["problem_type"] = _detect_problem_type(df_work[target])
        problem_type: str = state["problem_type"] or "classification"

        state["messages"].append(f"  Target: '{target}' | Task: {problem_type}")
        state["progress_pct"] = 12

        # ── 4. FINDINGS ───────────────────────────────────────────────────
        findings = await asyncio.to_thread(_generate_findings, df_work, profile["col_profiles"], target, problem_type)
        state["eda_findings"] = findings
        n_critical = sum(1 for f in findings if f["severity"] == "critical")
        state["messages"].append(f"  Found {len(findings)} findings ({n_critical} critical)")
        state["progress_pct"] = 15

        # ── 5. CHARTS ────────────────────────────────────────────────────
        charts, chart_warnings = await asyncio.to_thread(
            _generate_all_charts, df_work, target, problem_type, profile
        )
        state["warnings"].extend(chart_warnings)
        state["eda_charts"] = charts
        state["messages"].append(f"  Generated {len(charts)} EDA charts")
        state["progress_pct"] = 18

        # ── 6. LLM NARRATIVE ─────────────────────────────────────────────
        compact_profile = {
            "n_rows": profile["n_rows"],
            "n_cols": profile["n_cols"],
            "missing_pct": round(profile["missing_pct"], 3),
            "problem_type": problem_type,
            "target": target,
            "critical_findings": [f for f in findings if f["severity"] == "critical"][:5],
            "warnings": [f for f in findings if f["severity"] == "warning"][:5],
            "n_numeric": profile["n_numeric"],
            "n_categorical": profile["n_categorical"],
        }
        try:
            state["eda_narrative"] = await _call_llm_for_narrative(llm, compact_profile)
        except Exception as e:
            error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
            state["warnings"].append(f"LLM narrative generation failed: {error_msg[:100]}")
            state["eda_narrative"] = EDA_NARRATIVE_FALLBACK

        state["messages"].append("✓ EDA complete")
        state["progress_pct"] = 20

    except Exception as e:
        error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
        state["errors"].append(f"EDA agent: {error_msg}")
        state["messages"].append(f"⚠ EDA completed with errors: {error_msg[:100]}")
        # Initialize required fields to safe defaults
        if not state.get("profile"):
            state["profile"] = None
        if not state.get("eda_findings"):
            state["eda_findings"] = []
        if not state.get("eda_charts"):
            state["eda_charts"] = []
        if not state.get("eda_narrative"):
            state["eda_narrative"] = EDA_NARRATIVE_FALLBACK

    return state
