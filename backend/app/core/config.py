from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    STORAGE_PATH: str = "./data/storage"
    MAX_UPLOAD_SIZE_MB: int = 500
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/ml_intern.db"
    LARGE_DATASET_THRESHOLD: int = 100_000
    SAMPLE_SIZE_LARGE: int = 50_000
    MAX_TRAINING_TIMEOUT_SECONDS: int = 300
    MAX_COLS_FULL_ANALYSIS: int = 200

    class Config:
        env_file = ".env"


settings = Settings()
