# Setu — one container, one port, no build step for the front end.
#
# This image runs ASR through the LLM provider (SETU_ASR=gemini) rather than a
# local Whisper model. That is not the preferred architecture -- speech leaving
# the machine is a real cost, and setu/voice.py keeps the local path as the
# default for laptops and for the IVR stage.
#
# It is the architecture free hosting permits. A local model wants a real CPU
# and free tiers give you about a tenth of one, which turns a five-second
# transcription into fifty. Removing it drops the image from ~2GB to ~150MB and
# the build from ~10 minutes to under two, so this runs on the free tier of
# essentially anything.
#
# Measured, on the same Hindi clip: gemini 2.1s, whisper small 4.9s -- and the
# remote transcript was the more accurate of the two.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Managed hosts run containers as a non-root user. Create one and hand it the
# directories the app writes caches into, rather than finding out at the first
# request that it cannot.
RUN useradd -m -u 1000 setu \
    && mkdir -p /app/data/llm_cache /app/data/voice_cache \
    && chown -R setu:setu /app

USER setu
ENV HOME=/home/setu

COPY --chown=setu:setu . .

ENV SETU_ASR=gemini \
    PORT=7860

EXPOSE 7860

# GEMINI_API_KEY comes from the host's secret store, never baked in.
CMD ["sh", "-c", "uvicorn setu.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
