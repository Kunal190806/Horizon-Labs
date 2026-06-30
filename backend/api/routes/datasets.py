"""
backend/api/routes/datasets.py
Dataset management endpoints: search MAST, download TESS, upload files.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies import get_db
from backend.core.config import settings
from backend.models.orm import Dataset
from backend.models.schemas import DatasetOut, DatasetPreview, TESSSearchResult

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["Datasets"])


# ── Search MAST ──────────────────────────────────────────────────────────────
@router.get("/search", response_model=List[TESSSearchResult], summary="Search MAST for TESS observations")
async def search_mast(tic_id: str, sectors: Optional[int] = None):
    """Search the MAST archive for available TESS observations for a given TIC ID."""
    try:
        import lightkurve as lk
        results = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS")
        if len(results) == 0:
            return []

        out = []
        for i, row in enumerate(results.table):
            try:
                sector = int(row.get("sequence_number") or row.get("#") or i + 1)
                exptime = float(row.get("exptime") or row.get("t_exptime") or 0)
                description = str(row.get("description") or row.get("productSubGroupDescription") or "TESS LC")
                out.append(TESSSearchResult(
                    tic_id=tic_id,
                    sector=sector,
                    exptime=exptime,
                    mission="TESS",
                    description=description,
                ))
            except Exception:
                continue
        return out[:20]  # cap at 20 results
    except ImportError:
        raise HTTPException(503, "Lightkurve is not installed on this server.")
    except Exception as exc:
        logger.error("MAST search failed for TIC %s: %s", tic_id, exc)
        raise HTTPException(502, f"MAST search failed: {exc}")


# ── Download from MAST ────────────────────────────────────────────────────────
@router.post("/download", response_model=DatasetOut, summary="Download TESS light curve from MAST")
async def download_tess(
    tic_id: str,
    sector: Optional[int] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    db: AsyncSession = Depends(get_db),
):
    """Download a TESS light curve from MAST and store it locally."""
    try:
        import lightkurve as lk
        import numpy as np

        search_query = f"TIC {tic_id}"
        search_result = lk.search_lightcurve(search_query, mission="TESS", sector=sector)
        if len(search_result) == 0:
            raise HTTPException(404, f"No TESS data found for TIC {tic_id}" + (f" sector {sector}" if sector else ""))

        lc_collection = search_result[0].download(download_dir=settings.mast_cache_dir)
        lc = lc_collection.normalize()

        # Extract arrays
        time = lc.time.value.tolist()
        flux = lc.flux.value.tolist()
        flux_err = lc.flux_err.value.tolist() if hasattr(lc, "flux_err") and lc.flux_err is not None else None

        # Save as CSV
        os.makedirs(settings.datasets_dir, exist_ok=True)
        ds_id = str(uuid.uuid4())
        actual_sector = getattr(lc, "sector", sector or 0)
        fname = f"TIC{tic_id}_S{actual_sector}_{ds_id[:8]}.csv"
        fpath = os.path.join(settings.datasets_dir, fname)

        import csv
        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = ["time", "flux"]
            if flux_err:
                header.append("flux_err")
            writer.writerow(header)
            for i in range(len(time)):
                row = [time[i], flux[i]]
                if flux_err:
                    row.append(flux_err[i])
                writer.writerow(row)

        # Persist to DB
        dataset = Dataset(
            id=ds_id,
            name=f"TIC {tic_id} S{actual_sector}",
            tic_id=tic_id,
            source="mast",
            sector=int(actual_sector) if actual_sector else None,
            file_path=fpath,
            file_type="csv",
            num_points=len(time),
            time_start=float(time[0]) if time else None,
            time_end=float(time[-1]) if time else None,
        )
        db.add(dataset)
        await db.commit()
        await db.refresh(dataset)
        logger.info("Downloaded TIC %s sector %s: %d points", tic_id, actual_sector, len(time))
        return dataset

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Download failed for TIC %s: %s", tic_id, exc)
        raise HTTPException(500, f"Download failed: {exc}")


# ── Upload File ───────────────────────────────────────────────────────────────
@router.post("/upload", response_model=DatasetOut, summary="Upload a FITS or CSV light curve")
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    tic_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Upload a custom FITS or CSV light curve file."""
    allowed_types = {".fits", ".fit", ".csv", ".csv.gz", ".gz"}
    filename = file.filename or ""
    # Handle double extension like .csv.gz
    if filename.lower().endswith(".csv.gz"):
        suffix = ".csv.gz"
    else:
        suffix = Path(filename).suffix.lower()
    if suffix not in allowed_types:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: .fits, .fit, .csv, .csv.gz")

    content = await file.read()
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(413, f"File exceeds maximum size of {settings.max_upload_size_mb} MB.")

    os.makedirs(settings.datasets_dir, exist_ok=True)
    ds_id = str(uuid.uuid4())
    fname = f"{ds_id[:8]}_{file.filename}"
    fpath = os.path.join(settings.datasets_dir, fname)

    with open(fpath, "wb") as f:
        f.write(content)

    # Parse to extract metadata
    num_points, time_start, time_end = None, None, None
    try:
        if suffix in (".csv", ".csv.gz", ".gz"):
            import csv
            import gzip
            
            # Stream from disk instead of loading everything into memory
            if suffix in (".csv.gz", ".gz"):
                f_in = gzip.open(fpath, "rt", encoding="utf-8", errors="replace")
            else:
                f_in = open(fpath, "r", encoding="utf-8", errors="replace")
                
            with f_in as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    time_idx = next((i for i, c in enumerate(header) if c.lower() in ["time", "t", "bjd", "jd"]), 0)
                    
                    # Count points and track min/max time without storing the whole array
                    count = 0
                    first_time = None
                    last_time = None
                    
                    for row in reader:
                        if len(row) > time_idx:
                            try:
                                val = float(row[time_idx])
                                if count == 0:
                                    first_time = val
                                last_time = val
                                count += 1
                            except ValueError:
                                pass
                                
                    if count > 0:
                        num_points = count
                        time_start = first_time
                        time_end = last_time
        elif suffix in {".fits", ".fit"}:
            raise HTTPException(400, "FITS file parsing is disabled in Vercel Serverless environment due to bundle size limits. Please upload a CSV file.")
    except Exception as e:
        logger.warning("Could not parse metadata from %s: %s", fname, e)

    dataset = Dataset(
        id=ds_id,
        name=name,
        tic_id=tic_id,
        source="upload",
        file_path=fpath,
        file_type="csv" if suffix in (".csv", ".csv.gz", ".gz") else suffix.lstrip("."),
        num_points=num_points,
        time_start=time_start,
        time_end=time_end,
    )
    db.add(dataset)
    await db.commit()
    await db.refresh(dataset)
    logger.info("Uploaded dataset '%s' (%d bytes, %s points)", name, len(content), num_points)
    return dataset


# ── List All Datasets ─────────────────────────────────────────────────────────
@router.get("/", response_model=List[DatasetOut], summary="List all datasets")
async def list_datasets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).order_by(Dataset.created_at.desc()))
    return result.scalars().all()


# ── Get Single Dataset ────────────────────────────────────────────────────────
@router.get("/{dataset_id}", response_model=DatasetOut, summary="Get dataset metadata")
async def get_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, f"Dataset {dataset_id} not found.")
    return dataset


# ── Preview Dataset ────────────────────────────────────────────────────────────
@router.get("/{dataset_id}/preview", response_model=DatasetPreview, summary="Preview raw time/flux arrays")
async def preview_dataset(dataset_id: str, max_points: int = 5000, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    if not dataset.file_path or not os.path.exists(dataset.file_path):
        raise HTTPException(404, "Dataset file not found on disk.")

    try:
        import pandas as pd
        import numpy as np
        from astropy.io import fits

        ftype = (dataset.file_type or "").lower()
        fpath_lower = (dataset.file_path or "").lower()
        if ftype in ("fits", "fit"):
            raise HTTPException(400, "FITS previews are disabled in Vercel Serverless environment.")
        else:
            import csv
            import gzip
            import io
            is_gz = fpath_lower.endswith(".csv.gz") or fpath_lower.endswith(".gz")
            if is_gz:
                with gzip.open(dataset.file_path, "rt", encoding="utf-8") as f:
                    text = f.read()
                reader = csv.reader(io.StringIO(text))
            else:
                f = open(dataset.file_path, "r", encoding="utf-8")
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    raise ValueError("Empty CSV file")
                
                time_idx = next((i for i, c in enumerate(header) if c.lower() in ["time", "t", "bjd"]), 0)
                flux_idx = next((i for i, c in enumerate(header) if c.lower() in ["flux", "sap_flux", "pdcsap_flux"]), 1)
                err_idx = next((i for i, c in enumerate(header) if c.lower() in ["flux_err", "sap_flux_err"]), None)
                
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
                time = np.array(time_list)
                flux = np.array(flux_list)
                flux_err = np.array(err_list) if err_list else None

        # Remove NaN and downsample
        mask = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[mask], flux[mask]
        if flux_err is not None:
            flux_err = flux_err[mask]

        if len(time) > max_points:
            idx = np.round(np.linspace(0, len(time) - 1, max_points)).astype(int)
            time, flux = time[idx], flux[idx]
            flux_err = flux_err[idx] if flux_err is not None else None

        return DatasetPreview(
            id=dataset_id,
            name=dataset.name,
            time=time.tolist(),
            flux=flux.tolist(),
            flux_err=flux_err.tolist() if flux_err is not None else None,
            num_points=len(time),
        )
    except Exception as exc:
        logger.error("Preview failed for %s: %s", dataset_id, exc)
        raise HTTPException(500, f"Could not read dataset: {exc}")


# ── Delete Dataset ────────────────────────────────────────────────────────────
@router.delete("/{dataset_id}", summary="Delete a dataset")
async def delete_dataset(dataset_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(404, "Dataset not found.")
    if dataset.file_path and os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)
    await db.delete(dataset)
    await db.commit()
    return {"ok": True, "message": f"Dataset {dataset_id} deleted."}
