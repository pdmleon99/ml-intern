import os
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.core.config import settings
from app.core.storage import save_upload
from app.models.schemas import HuggingFaceRequest
from app.tools.dataset_tools import get_preview, load_dataframe

router = APIRouter()


@router.post("/datasets/upload")
async def upload_dataset(request: Request, file: UploadFile = File(...)):
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    allowed_extensions = {".csv", ".parquet", ".xlsx", ".xls", ".json", ".tsv", ".zip"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{suffix}'. Supported: {allowed_extensions}",
        )

    file_path = await save_upload(file)

    file_size = os.path.getsize(file_path)
    if file_size > max_bytes:
        os.remove(file_path)
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({file_size / 1e6:.1f}MB). Max: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    try:
        df = load_dataframe(file_path)
        preview = get_preview(df, n_rows=10)
        columns = list(df.columns)
        n_rows = len(df)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=422, detail=f"Could not read file: {str(e)[:200]}")

    return {
        "file_path": file_path,
        "filename": file.filename,
        "n_rows": n_rows,
        "columns": columns,
        "preview": preview,
    }


@router.post("/datasets/from-huggingface")
async def load_from_huggingface(payload: HuggingFaceRequest, request: Request):
    try:
        import uuid

        from datasets import load_dataset

        ds = load_dataset(payload.dataset_id, split=payload.split)
        df = ds.to_pandas()
        if len(df) > payload.max_rows:
            df = df.sample(n=payload.max_rows, random_state=42)

        os.makedirs(settings.STORAGE_PATH, exist_ok=True)
        fname = f"{uuid.uuid4().hex}_hf_{payload.dataset_id.replace('/', '_')}.parquet"
        file_path = os.path.join(settings.STORAGE_PATH, fname)
        df.to_parquet(file_path, index=False)

        preview = get_preview(df, n_rows=10)
        return {
            "file_path": file_path,
            "n_rows": len(df),
            "columns": list(df.columns),
            "preview": preview,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"HuggingFace load failed: {str(e)[:300]}")
