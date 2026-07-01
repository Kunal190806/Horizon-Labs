"""
backend/api/routes/preprocessing.py
Preprocessing endpoint: clean + detrend a dataset's light curve.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_db
from backend.models.orm import Dataset, ProcessingJob
from backend.models.schemas import PreprocessingConfig, PreprocessingResult
from backend.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/preprocessing", tags=["Preprocessing"])


async def _load_light_curve(dataset: Dataset):
    """Load time/flux arrays from a dataset file."""
    ftype = (dataset.file_type or "").lower()
    fpath = dataset.file_path

    if ftype in ("fits", "fit"):
        raise ValueError("FITS parsing is disabled in Vercel Serverless environment.")
    else:
        import csv
        import gzip
        
        is_gz = fpath.lower().endswith(".gz") or fpath.lower().endswith(".csv.gz")
        if is_gz:
            f = gzip.open(fpath, "rt", encoding="utf-8", errors="replace")
        else:
            f = open(fpath, "r", encoding="utf-8", errors="replace")

        with f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                raise ValueError("Empty CSV file")
            
            time_idx = next((i for i, c in enumerate(header) if c.lower() in ["time", "t", "bjd"]), 0)
            flux_idx = next((i for i, c in enumerate(header) if c.lower() in ["flux", "pdcsap_flux", "sap_flux"]), 1)
            err_idx = next((i for i, c in enumerate(header) if c.lower() in ["flux_err", "pdcsap_flux_err"]), None)
            
            time_list, flux_list, err_list = [], [], []
            for row in reader:
                if len(row) > max(time_idx, flux_idx):
                    try:
                        time_list.append(float(row[time_idx]))
                        flux_list.append(float(row[flux_idx]))
                        if err_idx is not None and len(row) > err_idx:
                            err_list.append(float(row[err_idx]))
                    except ValueError:
                        pass
            
            return np.array(time_list), np.array(flux_list), np.array(err_list) if err_list else None


@router.post("/{dataset_id}", response_model=PreprocessingResult, summary="Run preprocessing pipeline")
async def run_preprocessing(
    dataset_id: str,
    config: Optional[PreprocessingConfig] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Run the full preprocessing pipeline on a dataset.
    Returns raw and cleaned arrays plus summary statistics.
    """
    config = config or PreprocessingConfig()

    result_row = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result_row.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    if not dataset.file_path:
        raise HTTPException(400, "Dataset has no associated file.")

    # Create processing job
    job = ProcessingJob(
        id=str(uuid.uuid4()),
        dataset_id=dataset_id,
        job_type="preprocessing",
        status="running",
        started_at=datetime.utcnow(),
    )
    db.add(job)
    await db.commit()

    try:
        time, flux, flux_err = await _load_light_curve(dataset)

        pipeline = PreprocessingPipeline(
            sigma_clip_sigma=config.sigma_clip,
            detrend_method=config.detrend_method,
            savgol_window=config.savgol_window,
            interpolate=config.interpolate,
            normalize=config.normalize,
        )
        result = pipeline.run(time, flux, flux_err)

        # Update job
        job.status = "completed"
        job.progress = 100
        job.completed_at = datetime.utcnow()
        job.result = result.summary
        await db.commit()

        # Downsample for response (max 10k points)
        MAX_RESP = 10000

        def _downsample(arr: np.ndarray, n: int) -> list:
            if len(arr) <= n:
                return arr.tolist()
            idx = np.round(np.linspace(0, len(arr) - 1, n)).astype(int)
            return arr[idx].tolist()

        return PreprocessingResult(
            dataset_id=dataset_id,
            job_id=job.id,
            raw_time=_downsample(result.raw_time, MAX_RESP),
            raw_flux=_downsample(result.raw_flux, MAX_RESP),
            clean_time=_downsample(result.clean_time, MAX_RESP),
            clean_flux=_downsample(result.clean_flux, MAX_RESP),
            detrended_flux=_downsample(result.clean_flux, MAX_RESP),
            trend=_downsample(result.trend, min(MAX_RESP, len(result.trend))),
            outliers_removed=result.outliers_removed,
            gaps_interpolated=result.gaps_interpolated,
            summary=result.summary,
        )

    except Exception as exc:
        job.status = "failed"
        job.error = str(exc)
        job.completed_at = datetime.utcnow()
        await db.commit()
        logger.error("Preprocessing failed for %s: %s", dataset_id, exc)
        raise HTTPException(500, f"Preprocessing failed: {exc}")
