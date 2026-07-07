import os
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from .config import settings


def get_storage_path() -> Path:
    path = Path(settings.STORAGE_PATH)
    path.mkdir(parents=True, exist_ok=True)
    return path


async def save_upload(file: UploadFile) -> str:
    storage = get_storage_path()
    ext = Path(file.filename).suffix if file.filename else ".csv"
    filename = f"{uuid.uuid4().hex}{ext}"
    dest = storage / filename

    async with aiofiles.open(dest, "wb") as f:
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            await f.write(chunk)

    return str(dest)


def get_file_path(filename: str) -> Path:
    return get_storage_path() / filename


def ensure_data_dirs():
    os.makedirs("./data/storage", exist_ok=True)
    os.makedirs("./data/mlflow", exist_ok=True)
