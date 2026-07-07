from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler

from app.tools.dataset_tools import classify_string_column

from .state import AgentState


def _extract_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dt = pd.to_datetime(df[col], errors="coerce", infer_datetime_format=True)
    df = df.drop(columns=[col])
    df[f"{col}_year"] = dt.dt.year.fillna(0).astype(int)
    df[f"{col}_month"] = dt.dt.month.fillna(0).astype(int)
    df[f"{col}_dayofweek"] = dt.dt.dayofweek.fillna(0).astype(int)
    df[f"{col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
    min_date = dt.min()
    df[f"{col}_days_since_min"] = (dt - min_date).dt.days.fillna(0).astype(int)
    return df


def _try_prefix_extraction(series: pd.Series) -> Optional[pd.Series]:
    """Try to extract a low-cardinality prefix from a free-text column.
    Returns the extracted Series if 2 <= n_unique <= 20, else None.
    """
    for sep in [",", "."]:
        extracted = series.astype(str).str.split(sep).str[0].str.strip()
        n_u = extracted.replace("", float("nan")).nunique()
        if 2 <= n_u <= 20:
            return extracted
    return None


async def run_feature_agent(state: AgentState) -> AgentState:
    state["current_agent"] = "features"
    state["progress_pct"] = 22
    state["messages"].append("🔧 Engineering features...")

    try:
        sample_path = state.get("df_sample_path") or state["dataset_path"]
        try:
            df = pd.read_parquet(sample_path)
        except Exception:
            from app.tools.dataset_tools import load_dataframe
            df = load_dataframe(state["dataset_path"])

        features_applied = []
        dropped_columns = []
        target = state["target_column"]
        col_profiles = (state.get("profile") or {}).get("col_profiles", {})

        if not target or target not in df.columns:
            state["errors"].append(f"Target column '{target}' not found in dataset.")
            state["status"] = "failed"
            return state

        # ── 1. DROP TARGET NaN ROWS ──────────────────────────────────────
        n_before = len(df)
        df = df.dropna(subset=[target])
        if len(df) < n_before:
            state["messages"].append(f"  Dropped {n_before - len(df)} rows with NaN target")
        if len(df) < 10:
            state["errors"].append("Too few rows remain after dropping NaN targets.")
            state["status"] = "failed"
            return state

        # ── 1b. APPLY PLANNER'S COLUMN DROPS (agentic decision, not a hardcoded rule) ─
        plan = state.get("plan") or {}
        for decision in plan.get("columns_to_drop", []):
            col = decision.get("column")
            if not col or col == target or col not in df.columns:
                continue
            df = df.drop(columns=[col])
            dropped_columns.append({
                "column": col,
                "reason": f"[Planner] {decision.get('reason', 'flagged by planner')}",
                "dtype": "n/a",
                "n_unique": None,
            })
            state["messages"].append(f"  Planner dropped '{col}': {decision.get('reason', '')[:80]}")

        # ── 2. ENCODE STRING TARGET (deterministic — safe outside pipeline) ─
        if state["problem_type"] == "classification" and pd.api.types.is_object_dtype(df[target]):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[target] = le.fit_transform(df[target].astype(str))
            features_applied.append({
                "name": f"{target}_label_encoded",
                "type": "target_encoding",
                "rationale": "label encoding for string classification target",
                "columns_affected": [target],
            })

        # ── 3. DATETIME EXTRACTION (deterministic — safe outside pipeline) ─
        for col in list(df.columns):
            if col == target:
                continue
            profile = col_profiles.get(col, {})
            is_dt = (
                profile.get("is_datetime")
                or pd.api.types.is_datetime64_any_dtype(df[col])
            )
            if not is_dt and pd.api.types.is_object_dtype(df[col]):
                sample = df[col].dropna().head(100)
                converted = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
                is_dt = converted.notna().mean() > 0.7
            if is_dt:
                try:
                    df = _extract_datetime_features(df, col)
                    features_applied.append({
                        "name": f"{col}_datetime_features",
                        "type": "datetime_extraction",
                        "rationale": "extract year, month, day_of_week, is_weekend, days_since_min",
                        "columns_affected": [col],
                    })
                except Exception as e:
                    state["warnings"].append(f"Datetime extraction failed for '{col}': {str(e)[:60]}")

        # ── 4. CLASSIFY COLUMNS FOR PIPELINE ────────────────────────────
        numeric_cols: list[str] = []
        low_card_cat_cols: list[str] = []   # n_unique <= 10
        mid_card_cat_cols: list[str] = []   # 10 < n_unique <= 50
        cols_to_drop_set: set[str] = set()

        for col in list(df.columns):
            if col == target:
                continue

            n_unique = int(df[col].nunique())
            missing_pct = float(df[col].isnull().mean())
            dtype_str = str(df[col].dtype)

            def _drop(reason: str):
                cols_to_drop_set.add(col)
                dropped_columns.append({
                    "column": col,
                    "reason": reason,
                    "dtype": dtype_str,
                    "n_unique": n_unique,
                })

            # Constant or near-zero unique
            if n_unique <= 1:
                _drop("constant — zero variance")
                continue

            # Too many missing values
            if missing_pct > 0.4:
                _drop(f"{missing_pct:.0%} missing values")
                continue

            # Bool → treat as numeric
            if pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].astype(int)
                numeric_cols.append(col)
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                n_nonull = len(df[col].dropna())
                # ID columns are (near-)integer-valued; continuous measurements (age, price,
                # sqft...) are also often near-unique but have fractional values — only the
                # former should be dropped, or every continuous feature gets flagged as an ID.
                non_null_vals = df[col].dropna()
                is_integer_like = (
                    pd.api.types.is_integer_dtype(df[col])
                    or (len(non_null_vals) > 0 and (non_null_vals % 1 == 0).all())
                )
                if (n_nonull > 0 and is_integer_like
                        and n_unique / n_nonull > 0.95 and n_unique > 50):
                    _drop("numeric ID column — uniqueness > 95%")
                else:
                    numeric_cols.append(col)
                continue

            is_str = pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])
            if is_str:
                if n_unique <= 10:
                    low_card_cat_cols.append(col)
                elif n_unique <= 50:
                    mid_card_cat_cols.append(col)
                else:
                    # 3-way classifier for high-cardinality strings
                    str_type = classify_string_column(df[col])
                    if str_type == "id":
                        _drop("string ID column — near-unique values with no spaces")
                    elif str_type == "free_text":
                        extracted = _try_prefix_extraction(df[col])
                        if extracted is not None:
                            new_col = f"{col}_prefix"
                            df[new_col] = extracted
                            low_card_cat_cols.append(new_col)
                            features_applied.append({
                                "name": new_col,
                                "type": "text_prefix_extraction",
                                "rationale": f"extracted low-cardinality prefix from free-text column '{col}'",
                                "columns_affected": [col],
                            })
                            dropped_columns.append({
                                "column": col,
                                "reason": f"free text — replaced by prefix feature '{new_col}'",
                                "dtype": dtype_str,
                                "n_unique": n_unique,
                            })
                        else:
                            _drop("free text — no low-cardinality prefix extractable (tried comma and period splits)")
                        cols_to_drop_set.add(col)
                    else:
                        _drop(f"high-cardinality categorical ({n_unique} unique) — no viable encoding")
                continue

            # Anything else (complex dtypes)
            _drop(f"unsupported dtype: {dtype_str}")

        # ── 5. COLUMN AUDIT ─────────────────────────────────────────────
        accounted = (
            set(numeric_cols)
            | set(low_card_cat_cols)
            | set(mid_card_cat_cols)
            | cols_to_drop_set
            | {target}
        )
        unaccounted = set(df.columns) - accounted
        if unaccounted:
            state["warnings"].append(f"Unaccounted columns {list(unaccounted)} — dropping as safety measure")
            for col in unaccounted:
                cols_to_drop_set.add(col)
                dropped_columns.append({
                    "column": col,
                    "reason": f"unclassified {df[col].dtype} column — safety drop",
                    "dtype": str(df[col].dtype),
                    "n_unique": int(df[col].nunique()),
                })

        all_input_cols = numeric_cols + low_card_cat_cols + mid_card_cat_cols
        if len(all_input_cols) < 1:
            # Persist what was actually dropped/applied so far — otherwise this
            # diagnostic info is silently lost on the early-return failure path.
            state["dropped_columns"] = dropped_columns
            state["features_applied"] = features_applied
            state["errors"].append("No usable feature columns remain after classification.")
            state["status"] = "failed"
            return state

        # ── 6. BUILD UNFITTED SKLEARN PIPELINE ──────────────────────────
        transformers = []

        if numeric_cols:
            transformers.append(("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]), numeric_cols))
            features_applied.append({
                "name": "numeric_pipeline",
                "type": "pipeline",
                "rationale": (
                    f"median imputation + RobustScaler on {len(numeric_cols)} "
                    f"numeric columns — fitted PER FOLD inside cross-validation"
                ),
                "columns_affected": numeric_cols,
            })

        if low_card_cat_cols:
            transformers.append(("low_cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, drop="first"
                )),
            ]), low_card_cat_cols))
            features_applied.append({
                "name": "low_cardinality_ohe",
                "type": "pipeline",
                "rationale": (
                    f"mode imputation + OneHotEncoder (drop=first) on "
                    f"{len(low_card_cat_cols)} low-cardinality columns"
                ),
                "columns_affected": low_card_cat_cols,
            })

        if mid_card_cat_cols:
            transformers.append(("mid_cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )),
            ]), mid_card_cat_cols))
            features_applied.append({
                "name": "mid_cardinality_ordinal",
                "type": "pipeline",
                "rationale": (
                    f"mode imputation + OrdinalEncoder on "
                    f"{len(mid_card_cat_cols)} mid-cardinality columns"
                ),
                "columns_affected": mid_card_cat_cols,
            })

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )

        # ── 7. SAVE RAW DATAFRAME + UNFITTED PREPROCESSOR ───────────────
        base = Path(state["dataset_path"])
        processed_path = str(base.parent / (base.stem + "_processed.parquet"))
        preprocessor_path = str(base.parent / (base.stem + "_preprocessor.joblib"))

        df.to_parquet(processed_path, index=False)
        joblib.dump(preprocessor, preprocessor_path)

        state["processed_dataset_path"] = processed_path
        state["preprocessor_path"] = preprocessor_path
        state["feature_names"] = all_input_cols
        state["features_applied"] = features_applied
        state["dropped_columns"] = dropped_columns
        state["messages"].append(
            f"✓ Pipeline ready: {len(all_input_cols)} input features "
            f"({len(numeric_cols)} numeric, {len(low_card_cat_cols)} low-cat, "
            f"{len(mid_card_cat_cols)} mid-cat) | {len(dropped_columns)} columns dropped"
        )
        state["progress_pct"] = 40

    except Exception as e:
        error_msg = str(e).replace(state["llm_config"].get("api_key", ""), "[REDACTED]")
        state["errors"].append(f"Feature agent: {error_msg}")
        state["messages"].append(f"⚠ Feature engineering error: {error_msg[:100]}")

    return state
