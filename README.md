<div align="center">

# 🔭 Horizon — Exoplanet Detection & Characterization Platform

**Automated transit detection, candidate validation, and planetary characterization using NASA TESS data**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind-3-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*Built for the ISRO Hackathon — Exoplanet Science Track*

</div>

---

## ✨ What is Horizon?

Horizon is a production-grade web platform that automates the complete exoplanet detection pipeline — from raw TESS light curve data to confident planet candidates. It integrates established astronomical algorithms (TLS, BLS) with modern deep learning (CNN, LSTM, Transformer) to minimize false positives and surface the most promising transit signals.

### The Full Pipeline

```
TESS Archive (MAST)          User Upload
       │                     (CSV / CSV.GZ / FITS)
       └──────────┬──────────┘
                  ▼
         ┌─────────────────┐
         │  Preprocessing  │  NaN removal → sigma-clip → detrend → gap-fill → normalize
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │ Transit Search  │  Transit Least Squares (TLS) + Box Least Squares (BLS)
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │   Validation    │  CNN/LSTM/Transformer + statistical tests + SHAP/LIME
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │Characterization │  Period, depth, radius, semi-major axis, T_eq
         └────────┬────────┘
                  ▼
         ┌─────────────────┐
         │    Reports      │  PDF + CSV export
         └─────────────────┘
```

---

## 🚀 Features

### 🛰️ Data Acquisition
- **MAST Archive search** — Find TESS observations by TIC ID, browse all available sectors
- **One-click download** — Pull TESS light curves directly from MAST via Lightkurve
- **Flexible upload** — Drag-and-drop `.csv`, `.csv.gz` (compressed), and `.fits` files
- **Live dataset preview** — Inspect time/flux arrays before processing

### ⚙️ Preprocessing Pipeline
| Step | Method |
|---|---|
| Invalid sample removal | NaN / Inf masking |
| Outlier rejection | Iterative σ-clipping (pure NumPy) + IQR |
| Detrending | Savitzky-Golay · Cubic spline · Wotan biweight |
| Gap interpolation | Linear fill of short cadence gaps |
| Normalization | Median normalization to unit flux |

### 🔍 Transit Detection
- **TLS** (Transit Least Squares) — gold standard for exoplanet transits
- **BLS** (Box Least Squares) — fast SciPy-based periodogram
- Extracted parameters: period, epoch, duration, depth, SNR, SDE, transit times
- Exports periodogram and phase-folded light curve data for visualization

### 🤖 Candidate Validation
**Statistical tests:**
- Odd-even transit depth comparison
- Transit shape consistency (trapezoid model fit)
- Depth stability across transits (CV test)
- Transit duration consistency (Kepler's 3rd Law / Seager & Mallén-Ornelas 2003)
- SNR threshold test

**ML classifiers** (CNN default, LSTM and Transformer available):
- Input: 201-bin phase-folded light curve
- Output: `PLANET` / `FALSE_POSITIVE` / `UNKNOWN` + confidence score
- Explainability: SHAP (DeepExplainer) → LIME segment-perturbation fallback

### 🪐 Planet Characterization
- Planet radius (R⊕ and R♃)
- Semi-major axis (AU)
- Equilibrium temperature (K)
- Classification: Hot Jupiter / Neptune-class / Super-Earth / Earth-like
- Stellar parameters fetched from MAST/Simbad when available

### 📊 Visualization Dashboard
Interactive Plotly charts for:
- Raw vs. cleaned light curve comparison
- Detrending result overlay
- Periodogram (TLS + BLS)
- Phase-folded transit with model fit
- Confidence gauge per candidate
- Multi-candidate parameter comparison

### 📄 Reports
- **PDF** — Scientific report with preprocessing summary, candidate table, parameters, and timestamps (ReportLab)
- **CSV** — Machine-readable export with all candidate fields

---

## 🏗️ Architecture

```
horizon/
├── backend/                  # FastAPI backend
│   ├── api/routes/           # REST endpoints (datasets, preprocessing, detection, …)
│   ├── preprocessing/        # NaN removal, outlier clip, detrending, gap fill
│   ├── transit_detection/    # TLS + BLS detectors
│   ├── candidate_validation/ # CNN / LSTM / Transformer + statistical tests
│   ├── characterization/     # Planet & stellar parameter estimation
│   ├── visualization/        # Server-side Plotly JSON generators
│   ├── reports/              # PDF + CSV report generation
│   ├── models/               # SQLAlchemy ORM + Pydantic schemas
│   ├── core/                 # Config, DB, Redis cache, logging
│   └── utils/                # Array helpers, time system conversions
├── frontend/                 # React 18 + Vite + Tailwind
│   └── src/
│       ├── pages/            # 8 pages (Dashboard → Reports)
│       ├── components/       # Layout, charts, custom cursor, 3D background
│       ├── api/              # Axios client wrappers
│       └── store/            # Zustand state management
├── tests/                    # Pytest test suite
├── docs/                     # Full technical documentation
└── docker/                   # Dockerfiles + Compose
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, FastAPI, SQLAlchemy (async), Alembic |
| **Science** | NumPy, SciPy, Astropy, Lightkurve, Astroquery |
| **ML** | TensorFlow/Keras (CNN, BiLSTM, Transformer), scikit-learn, SHAP |
| **Database** | PostgreSQL (production) · SQLite (local dev) |
| **Cache** | Redis |
| **Frontend** | React 18, Vite, Tailwind CSS, Plotly.js, Framer Motion |
| **Reports** | ReportLab (PDF) |
| **Infra** | Docker, Docker Compose |

---

## ⚡ Quick Start

### Option A — Local Development

```bash
# Clone
git clone https://github.com/Kunal190806/Horizon-Labs.git
cd Horizon-Labs

# Backend setup
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

pip install -r backend/requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: set USE_SQLITE=true, USE_REDIS=false  (for local dev without Docker)

# Start backend  (Windows)
set PYTHONIOENCODING=utf-8
python backend/main.py

# Start frontend  (new terminal)
cd frontend
npm install
npm run dev
```

| Service | URL |
|---|---|
| **Frontend** | http://localhost:5173 |
| **API** | http://localhost:8000 |
| **Swagger Docs** | http://localhost:8000/docs |

### Option B — Docker (Recommended)

```bash
docker compose up --build
```

Starts PostgreSQL, Redis, backend, and frontend in one command.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/datasets/` | List all datasets |
| `POST` | `/api/datasets/upload` | Upload CSV / CSV.GZ / FITS |
| `GET` | `/api/datasets/search?tic_id=` | Search MAST archive |
| `POST` | `/api/datasets/download` | Download from MAST |
| `GET` | `/api/datasets/{id}/preview` | Preview flux arrays |
| `POST` | `/api/preprocessing/run` | Run preprocessing pipeline |
| `POST` | `/api/detection/run` | Run TLS + BLS detection |
| `POST` | `/api/validation/run` | Run ML + statistical validation |
| `POST` | `/api/characterization/run` | Estimate planetary parameters |
| `GET` | `/api/visualization/{id}` | Fetch plot data |
| `POST` | `/api/reports/generate` | Generate PDF + CSV report |
| `WS` | `/ws/jobs/{job_id}` | Real-time job progress |

---

## 🧪 Running Tests

```bash
# Activate venv first
pytest tests/ -v
```

Test coverage: preprocessing pipeline, TLS/BLS detection, candidate validation.

---

## 🤝 Contributing

This project is part of an ISRO Hackathon submission. Contributions and improvements are welcome!

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push and open a Pull Request

---

## 📚 Documentation

Full technical documentation (architecture, pipeline usage, ML training, deployment guide, and extension guide for new missions like PLATO / Kepler / future ISRO missions) is available in [`docs/README.md`](docs/README.md).

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Made with ❤️ for the ISRO Hackathon · Powered by NASA TESS Data

</div>
