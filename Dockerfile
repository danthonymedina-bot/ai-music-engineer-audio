# Slim default image for Render FREE (512 MB RAM): no torch. Separation = Open-Unmix int8 ONNX (models/umxhq) -> DSP fallback.
FROM python:3.11-slim
ARG WITH_DEMUCS=0
ENV PYTHONUNBUFFERED=1 PORT=8000 \
    AUDIO_ENGINE_STORAGE=/tmp/audio-engine SEPARATION_BACKEND=auto \
    MALLOC_ARENA_MAX=2 MALLOC_TRIM_THRESHOLD_=65536 OMP_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/numba
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg rubberband-cli libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# Optional (paid plans with >=2 GB RAM only): docker build --build-arg WITH_DEMUCS=1
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-ml.txt && \
      python -c "from demucs.pretrained import get_model; get_model('htdemucs')" ; fi
COPY models ./models
COPY app ./app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
