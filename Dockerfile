# ─── Stage 1: build the React frontend ──────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --prefer-offline

COPY frontend/ ./

# Prevent CRA from treating warnings as errors in CI environments
ENV CI=false
RUN npm run build

# ─── Stage 2: Python runtime ─────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

WORKDIR /app

COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

# Pull in the compiled React assets from stage 1
COPY --from=frontend-build /build/build ./frontend/build/

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
