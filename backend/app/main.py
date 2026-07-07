from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.middleware.api_key import extract_llm_config
from app.api.routes import datasets, jobs, reports
from app.core.database import init_db
from app.core.storage import ensure_data_dirs


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_data_dirs()
    await init_db()
    yield


app = FastAPI(title="ML Intern", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(extract_llm_config)

app.include_router(jobs.router, prefix="/api")
app.include_router(datasets.router, prefix="/api")
app.include_router(reports.router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
