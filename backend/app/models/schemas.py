from typing import Optional

from pydantic import BaseModel


class CreateJobRequest(BaseModel):
    dataset_source: str = "upload"  # upload | huggingface | kaggle
    uploaded_file_path: Optional[str] = None
    hf_dataset_id: Optional[str] = None
    hf_split: str = "train"
    hf_max_rows: int = 500_000
    kaggle_dataset: Optional[str] = None
    kaggle_filename: Optional[str] = None
    user_description: str
    target_column: Optional[str] = None


class JobSummary(BaseModel):
    job_id: str
    status: str
    progress_pct: int
    created_at: Optional[str]
    dataset_name: str
    best_model_score: Optional[float]
    problem_type: Optional[str]


class HuggingFaceRequest(BaseModel):
    dataset_id: str
    split: str = "train"
    max_rows: int = 500_000


class KaggleRequest(BaseModel):
    dataset: str
    filename: str
