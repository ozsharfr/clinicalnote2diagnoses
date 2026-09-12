# ClinicalNote2Diagnoses — build downloads deps + mE5 + trains; runtime is fully offline.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hfcache \
    PIP_NO_CACHE_DIR=1

# Deps: CPU-only torch to keep the image small.
COPY requirements.txt .
RUN pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Source + data (CSVs are needed for build-time training).
COPY src/ ./src/
COPY api/ ./api/
COPY visits.csv physician_annotations.csv ./

# BUILD-TIME: download mE5 (internet allowed here) + fit model -> artifacts/model.pkl.
# The mE5 weights land in HF_HOME and the fitted model in ./artifacts, both baked in.
RUN python -m src.train

# RUNTIME: fully offline. These env vars force transformers/hub to use the baked cache.
ENV TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
