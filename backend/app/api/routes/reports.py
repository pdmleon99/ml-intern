import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.job import Job

router = APIRouter()


@router.get("/reports/{job_id}/pdf")
async def download_pdf(job_id: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.report_pdf_path or not os.path.exists(job.report_pdf_path):
        raise HTTPException(status_code=404, detail="Report PDF not yet available")

    return FileResponse(
        path=job.report_pdf_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=ml_intern_report_{job_id[:8]}.pdf"},
    )


@router.get("/reports/{job_id}/json")
async def get_report_json(job_id: str, db: AsyncSession = Depends(get_db)):
    from sqlalchemy import select
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if not job.report_json:
        raise HTTPException(status_code=404, detail="Report JSON not yet available")
    return job.report_json
