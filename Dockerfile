# ─── Stage 1: Build the React Frontend ───
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# ─── Stage 2: Create the FastAPI Backend Server ───
FROM python:3.11-slim

# System dependencies for scientific libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran libffi-dev libssl-dev curl \
    libhdf5-dev libnetcdf-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY backend/ ./backend/

# Copy compiled React frontend assets from Stage 1
COPY --from=frontend-builder /frontend/dist ./frontend/dist

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Expose default port (Render will override this dynamically using PORT env)
EXPOSE 10000

# Start command
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
