import uuid

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String, Text
from sqlalchemy.sql import func

from app.core.database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String, default="queued")
    progress_pct = Column(Integer, default=0)
    current_agent = Column(String, default="")
    dataset_name = Column(String, default="")
    dataset_path = Column(String, default="")
    user_description = Column(Text, default="")
    target_column = Column(String, nullable=True)
    problem_type = Column(String, nullable=True)
    best_model_name = Column(String, nullable=True)
    best_model_score = Column(Float, nullable=True)
    report_pdf_path = Column(String, nullable=True)
    messages = Column(JSON, default=list)
    errors = Column(JSON, default=list)
    warnings = Column(JSON, default=list)
    report_json = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
