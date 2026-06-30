import axios from 'axios'
import toast from 'react-hot-toast'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

const api = axios.create({
  baseURL: API_BASE,
  timeout: 300_000, // 5 min for long downloads
  headers: { 'Content-Type': 'application/json' },
})

// Response interceptor — global error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    const msg = error.response?.data?.detail || error.message || 'Request failed'
    if (error.response?.status !== 404) {
      toast.error(String(msg).slice(0, 120))
    }
    return Promise.reject(error)
  }
)

// ── Dataset APIs ──────────────────────────────────────────────────────────────
export const datasetsApi = {
  search: (ticId) => api.get(`/api/datasets/search`, { params: { tic_id: ticId } }),
  download: (ticId, sector) => api.post(`/api/datasets/download`, null, { params: { tic_id: ticId, sector } }),
  upload: (formData) => api.post(`/api/datasets/upload`, formData, { headers: { 'Content-Type': 'multipart/form-data' } }),
  list: () => api.get(`/api/datasets/`),
  get: (id) => api.get(`/api/datasets/${id}`),
  preview: (id, maxPoints = 5000) => api.get(`/api/datasets/${id}/preview`, { params: { max_points: maxPoints } }),
  delete: (id) => api.delete(`/api/datasets/${id}`),
}

// ── Preprocessing APIs ────────────────────────────────────────────────────────
export const preprocessingApi = {
  run: (datasetId, config = {}) => api.post(`/api/preprocessing/${datasetId}`, config),
}

// ── Detection APIs ────────────────────────────────────────────────────────────
export const detectionApi = {
  run: (datasetId, config = {}) => api.post(`/api/detection/${datasetId}`, config),
  listCandidates: (datasetId) => api.get(`/api/detection/${datasetId}/candidates`),
}

// ── Validation APIs ───────────────────────────────────────────────────────────
export const validationApi = {
  run: (candidateId) => api.post(`/api/validation/${candidateId}`),
  get: (candidateId) => api.get(`/api/validation/${candidateId}`),
}

// ── Characterization APIs ─────────────────────────────────────────────────────
export const characterizationApi = {
  run: (candidateId) => api.post(`/api/characterization/${candidateId}`),
  get: (candidateId) => api.get(`/api/characterization/${candidateId}`),
}

// ── Visualization APIs ────────────────────────────────────────────────────────
export const visualizationApi = {
  getPlots: (datasetId) => api.get(`/api/visualization/${datasetId}`),
}

// ── Reports APIs ──────────────────────────────────────────────────────────────
export const reportsApi = {
  generate: (payload) => api.post(`/api/reports/generate`, payload),
  downloadUrl: (filename) => `${API_BASE}/api/reports/download/${filename}`,
}

export default api
