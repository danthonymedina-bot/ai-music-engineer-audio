FROM python:3.11-slim
ARG WITH_DEMUCS=1
ENV PYTHONUNBUFFERED=1 AUDIO_ENGINE_STORAGE=/data PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg rubberband-cli libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# CPU-only torch keeps the image ~1.5 GB smaller than the default CUDA wheel
RUN if [ "$WITH_DEMUCS" = "1" ]; then \
      pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu && \
      pip install --no-cache-dir -r requirements-ml.txt && \
      python -c "from demucs.pretrained import get_model; get_model('htdemucs')" ; fi
COPY app ./app
RUN mkdir -p /data
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
