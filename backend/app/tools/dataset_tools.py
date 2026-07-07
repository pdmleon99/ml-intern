import zipfile
from pathlib import Path

import chardet
import pandas as pd


def detect_encoding(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read(100_000)
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


def load_dataframe(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".zip":
        with zipfile.ZipFile(file_path) as zf:
            csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_files:
                raise ValueError("No CSV files found inside the ZIP archive.")
            with zf.open(csv_files[0]) as f:
                encoding = "utf-8"
                return pd.read_csv(f, encoding=encoding, low_memory=False)

    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        encoding = detect_encoding(file_path)
        try:
            return pd.read_csv(file_path, sep=sep, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            return pd.read_csv(file_path, sep=sep, encoding="latin-1", low_memory=False)

    if suffix == ".parquet":
        return pd.read_parquet(file_path)

    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(file_path)

    if suffix == ".json":
        return pd.read_json(file_path)

    raise ValueError(f"Unsupported file format: {suffix}")


def count_rows(file_path: str) -> int:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        sep = "\t" if suffix == ".tsv" else ","
        encoding = detect_encoding(file_path)
        count = 0
        for _ in pd.read_csv(file_path, sep=sep, encoding=encoding, chunksize=10_000):
            count += len(_)
        return count
    df = load_dataframe(file_path)
    return len(df)


def classify_string_column(series: pd.Series) -> str:
    """3-way classifier for high-cardinality string columns (n_unique > 50).
    Returns: 'id' | 'free_text' | 'high_cardinality_categorical'
    """
    n = len(series.dropna())
    if n == 0:
        return "id"
    uniqueness = series.nunique() / n
    sample = series.dropna().sample(min(500, n), random_state=42)
    avg_words = sample.astype(str).apply(lambda x: len(x.split())).mean()
    median_chars = float(sample.astype(str).str.len().median())
    looks_numeric = sample.astype(str).str.match(r"^\s*[\d\-\.]+\s*$").mean() > 0.8
    if uniqueness > 0.95 and (looks_numeric or avg_words <= 1.2):
        return "id"
    elif avg_words > 2.5 or median_chars > 40:
        return "free_text"
    else:
        return "high_cardinality_categorical"


def infer_column_types(df: pd.DataFrame) -> dict:
    col_info = {}
    for col in df.columns:
        series = df[col]
        info = {
            "dtype": str(series.dtype),
            "n_unique": int(series.nunique()),
            "missing_count": int(series.isnull().sum()),
            "missing_pct": float(series.isnull().mean()),
            "is_constant": series.nunique() == 1,
            "is_id": False,
            "is_datetime": False,
            "is_free_text": False,
            "is_boolean_like": False,
        }

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if set(non_null.unique()).issubset({0, 1}) and len(non_null) > 0:
                info["is_boolean_like"] = True
            n = len(non_null)
            if n > 0:
                uniq_ratio = series.nunique() / n
                if uniq_ratio > 0.95 and series.nunique() > 50:
                    info["is_id"] = True

        if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
            non_null = series.dropna().astype(str)
            if len(non_null) > 0:
                uniq_ratio = series.nunique() / len(non_null)
                if series.nunique() > 50:
                    string_type = classify_string_column(series)
                    info["string_column_type"] = string_type
                    if string_type == "id":
                        info["is_id"] = True
                    elif string_type == "free_text":
                        info["is_free_text"] = True
                else:
                    # Low/mid cardinality — not id or free text
                    info["string_column_type"] = "categorical"

            # Try datetime conversion
            sample = non_null.head(200)
            converted = pd.to_datetime(sample, errors="coerce", infer_datetime_format=True)
            if converted.notna().mean() > 0.7:
                info["is_datetime"] = True

        if pd.api.types.is_datetime64_any_dtype(series):
            info["is_datetime"] = True

        col_info[col] = info
    return col_info


def profile_column(series: pd.Series, col_info: dict) -> dict:
    profile = dict(col_info)
    profile["n_unique"] = int(series.nunique())
    profile["missing_count"] = int(series.isnull().sum())
    profile["missing_pct"] = float(series.isnull().mean())

    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 0:
            profile.update({
                "mean": float(non_null.mean()),
                "median": float(non_null.median()),
                "std": float(non_null.std()),
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "p25": float(non_null.quantile(0.25)),
                "p75": float(non_null.quantile(0.75)),
                "p95": float(non_null.quantile(0.95)),
                "skewness": float(non_null.skew()),
                "kurtosis": float(non_null.kurtosis()),
                "n_zeros": int((non_null == 0).sum()),
                "n_negatives": int((non_null < 0).sum()),
                "near_zero_variance": float(non_null.std()) < 1e-10,
            })

    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
        vc = series.value_counts().head(10)
        profile["top_10_values"] = vc.to_dict()
        profile["is_high_cardinality"] = series.nunique() > 100

    elif profile.get("is_datetime") or pd.api.types.is_datetime64_any_dtype(series):
        dt = pd.to_datetime(series, errors="coerce")
        non_null = dt.dropna()
        if len(non_null) > 0:
            profile["min_date"] = str(non_null.min())
            profile["max_date"] = str(non_null.max())
            profile["range_days"] = int((non_null.max() - non_null.min()).days)

    return profile


def build_dataset_profile(df: pd.DataFrame, original_n_rows: int, is_large: bool) -> dict:
    col_types = infer_column_types(df)
    col_profiles = {}
    for col in df.columns:
        col_profiles[col] = profile_column(df[col], col_types[col])

    n_numeric = sum(1 for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))
    n_categorical = sum(1 for c in df.columns
                        if pd.api.types.is_object_dtype(df[c])
                        or pd.api.types.is_string_dtype(df[c]))
    dt_cols = [c for c in df.columns if col_types[c].get("is_datetime")]
    n_datetime = len(dt_cols)
    dup_rows = int(df.duplicated().sum())

    return {
        "n_rows": original_n_rows,
        "n_cols": len(df.columns),
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "n_datetime": n_datetime,
        "missing_pct": float(df.isnull().mean().mean()),
        "duplicate_rows": dup_rows,
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
        "col_profiles": col_profiles,
        "is_large": is_large,
        "sample_size": len(df),
    }


def get_preview(df: pd.DataFrame, n_rows: int = 10) -> list:
    preview = [list(df.columns)]
    for _, row in df.head(n_rows).iterrows():
        preview.append([str(v) if not pd.isna(v) else "" for v in row])
    return preview


def save_sample_parquet(df: pd.DataFrame, base_path: str) -> str:
    out = Path(base_path).with_suffix("") .parent / (Path(base_path).stem + "_sample.parquet")
    df.to_parquet(str(out), index=False)
    return str(out)
