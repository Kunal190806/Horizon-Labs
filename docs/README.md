# Horizon Exoplanet Platform — Documentation

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Pipeline Modules](#pipeline-modules)
5. [ML Models](#ml-models)
6. [Deployment](#deployment)
7. [Extending for New Missions](#extending-for-new-missions)

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────┐
│                    React Frontend                     │
│  (Vite + Tailwind + Plotly + Framer Motion)           │
│  Pages: Dashboard, Datasets, Preprocessing,           │
│         Detection, Validation, Characterization,      │
│         Visualization, Reports                        │
└───────────────────┬───────────────────────────────────┘
                    │ REST + WebSocket
┌───────────────────▼───────────────────────────────────┐
│                  FastAPI Backend                       │
│                                                        │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │ Datasets │ │Preprocess│ │Detection │ │Validatn │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘  │
│       │            │            │             │        │
│  ┌────▼────────────▼────────────▼─────────────▼────┐  │
│  │       Core Pipeline (NumPy / SciPy / Astropy)   │  │
│  └─────────────────────────────────────────────────┘  │
│                                                        │
│  ┌──────────────┐ ┌─────────┐ ┌───────┐               │
│  │ Characterize │ │ Reports │ │ Plots │               │
│  └──────────────┘ └─────────┘ └───────┘               │
└───────────────────┬───────────────────────────────────┘
                    │
     ┌──────────────┼──────────────┐
     ▼              ▼              ▼
 PostgreSQL       Redis         Local FS
 (metadata)     (cache)       (datasets/
                               reports)
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker + Docker Compose (optional, recommended)

### Local Development (no Docker)

```bash
# 1. Clone
git clone https://github.com/Kunal190806/Horizon-Labs.git
cd Horizon-Labs

# 2. Backend
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS

pip install -r backend/requirements.txt

# 3. Environment
cp .env.example .env
# Edit .env: set USE_SQLITE=true, USE_REDIS=false for local dev

# 4. Start backend
set PYTHONIOENCODING=utf-8      # Windows
export PYTHONIOENCODING=utf-8   # Linux/macOS
python backend/main.py

# 5. Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

The app will be at **http://localhost:5173** and the API at **http://localhost:8000/docs**.

### Docker (Recommended)

```bash
docker compose up --build
```

Services:
| Service | Port |
|---|---|
| Frontend | 3000 |
| Backend API | 8000 |
| PostgreSQL | 5432 |
| Redis | 6379 |

---

## API Reference

All endpoints are documented interactively at **http://localhost:8000/docs** (Swagger UI).

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/datasets/` | List all datasets |
| `POST` | `/api/datasets/upload` | Upload FITS / CSV / CSV.GZ |
| `GET` | `/api/datasets/search?tic_id=` | Search MAST archive |
| `POST` | `/api/datasets/download` | Download from MAST |
| `GET` | `/api/datasets/{id}/preview` | Preview time/flux arrays |
| `DELETE` | `/api/datasets/{id}` | Delete dataset |
| `POST` | `/api/preprocessing/run` | Run preprocessing pipeline |
| `GET` | `/api/preprocessing/status/{job_id}` | Poll job status |
| `POST` | `/api/detection/run` | Run TLS + BLS detection |
| `POST` | `/api/validation/run` | Run ML + statistical validation |
| `POST` | `/api/characterization/run` | Estimate planetary parameters |
| `GET` | `/api/visualization/{dataset_id}` | Fetch plot data |
| `POST` | `/api/reports/generate` | Generate PDF + CSV reports |
| `GET` | `/api/reports/download/{filename}` | Download report file |
| `WS` | `/ws/jobs/{job_id}` | Real-time job progress |

---

## Pipeline Modules

### 1. Preprocessing (`backend/preprocessing/`)

```python
from backend.preprocessing.pipeline import PreprocessingPipeline

pipeline = PreprocessingPipeline(
    sigma_clip_sigma=4.0,
    detrend_method="savgol",   # "savgol" | "spline" | "wotan"
    savgol_window=51,
    interpolate=True,
    normalize=True,
)
result = pipeline.run(time, flux, flux_err)
# result.clean_time, result.clean_flux, result.summary
```

Steps executed in order:
1. **NaN/Inf removal** — removes invalid samples
2. **Sigma-clip outlier removal** — iterative 4σ clipping
3. **Detrending** — Savitzky-Golay (default), spline, or Wotan biweight
4. **Gap interpolation** — linear fill of short cadence gaps
5. **Median normalization** — scale flux to unit median

### 2. Transit Detection (`backend/transit_detection/`)

```python
from backend.transit_detection.tls_detector import run_tls
from backend.transit_detection.bls_detector import run_bls

# TLS (uses transitleastsquares if installed, falls back to BLS)
result = run_tls(time, flux, flux_err, min_period=0.5, max_period=27.0, snr_threshold=7.0)

# BLS (uses scipy.signal.lombscargle internally)
result = run_bls(time, flux, flux_err, min_period=0.5, max_period=27.0)
```

Returned fields: `period`, `epoch`, `duration`, `depth`, `snr`, `sde`, `num_transits`, `periodogram_periods/power`, `folded_time/flux`.

### 3. Candidate Validation (`backend/candidate_validation/`)

```python
from backend.candidate_validation.statistical_validator import validate_candidate
from backend.candidate_validation.ml_classifier import get_classifier

# Statistical tests (odd-even, shape, depth stability, duration consistency, SNR)
stat = validate_candidate(time, flux, period, epoch, duration, depth, snr)
# stat.score, stat.odd_even_flag, stat.duration_flag, ...

# ML classifier (CNN / LSTM / Transformer)
clf = get_classifier(model_type="cnn")   # "cnn" | "lstm" | "transformer"
pred = clf.predict(folded_time, folded_flux)
# pred.label ("PLANET" | "FALSE_POSITIVE" | "UNKNOWN")
# pred.confidence, pred.planet_prob, pred.fp_prob
# pred.shap_values, pred.lime_explanation
```

### 4. Characterization (`backend/characterization/`)

```python
from backend.characterization.planet_params import characterize_planet

params = characterize_planet(
    period=3.14, depth=0.01, duration=0.1,
    stellar_radius=1.0, stellar_mass=1.0, stellar_temp=5778.0
)
# params.planet_radius_rearth, params.semi_major_axis_au,
# params.equilibrium_temp_k, params.classification
```

---

## ML Models

### Architecture Comparison

| Architecture | Params | Strengths | Best For |
|---|---|---|---|
| **CNN** (default) | ~50K | Fast, translation-invariant | Box-like transits |
| **BiLSTM** | ~120K | Sequential context | Long-duration, complex patterns |
| **Transformer** | ~80K | Global attention, interpretable | Multi-harmonic, irregular cadence |

All models:
- Input: 201-bin phase-folded light curve
- Output: softmax `[FP_prob, Planet_prob]`
- Trained on: synthetic transit + false-positive dataset (Astronet-Triage style)
- Explainability: SHAP (preferred) → LIME segment-perturbation (fallback)

### Training

```bash
# Train CNN model (requires TensorFlow)
python -c "
from backend.candidate_validation.model_training import train_model
metrics = train_model('backend/ml_models/transit_cnn.weights.h5', epochs=50)
print(metrics)
"
```

Callbacks active during training:
- `EarlyStopping` (patience=10, `val_accuracy`)
- `ReduceLROnPlateau` (patience=5, factor=0.5)
- `ModelCheckpoint` (save best weights)
- `TensorBoard` (logs at `backend/ml_models/logs/`)

### Evaluation Metrics

After training, the following metrics are returned:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- **ROC-AUC**
- **Average Precision** (area under PR curve)
- **MCC** (Matthews Correlation Coefficient)
- **Confusion Matrix**
- **Full Precision-Recall Curve** (precision[], recall[], thresholds[])

---

## Deployment

### Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `USE_SQLITE` | `false` | Use SQLite instead of PostgreSQL |
| `USE_REDIS` | `true` | Enable Redis caching |
| `DATABASE_URL` | — | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL |
| `MAX_UPLOAD_SIZE_MB` | `500` | Maximum upload file size |
| `DATASETS_DIR` | `backend/datasets` | Local dataset storage path |
| `REPORTS_DIR` | `backend/reports/output` | Report output directory |
| `CNN_MODEL_PATH` | `backend/ml_models/transit_cnn.weights.h5` | Trained model weights |
| `APP_ENV` | `development` | `development` / `production` |

### Production

```bash
# Build and start all services
docker compose -f docker-compose.yml up -d --build

# Apply DB migrations
docker compose exec backend alembic upgrade head
```

---

## Extending for New Missions

The platform is designed for easy extension:

### Adding a New Survey (e.g., PLATO, Kepler)

1. **Data acquisition**: Add a new download function in `backend/api/routes/datasets.py` following the `download_tess` pattern. PLATO data can be fetched via ESA's PLATO archive API.

2. **File format support**: Add the new file format handler in the upload endpoint (currently supports FITS, CSV, CSV.GZ).

3. **Time system**: Add conversion functions to `backend/utils/time_utils.py` (e.g., `plato_bjd_to_bjd()`).

4. **Pipeline parameters**: PLATO's cadence differs from TESS. Adjust `gap_threshold_days` and `savgol_window` in `PreprocessingPipeline` for the new mission's cadence.

5. **Frontend**: Add the new mission as a search source option in `frontend/src/pages/Datasets.jsx`.

No changes are required to the detection, validation, or characterization modules — they are agnostic to the data source.
