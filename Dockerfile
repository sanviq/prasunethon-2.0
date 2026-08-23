# Setu — one container, one port, no build step for the front end.
#
# Python 3.12 rather than 3.14: ctranslate2 (what faster-whisper runs on) ships
# manylinux wheels a release or two behind, and a host that has to compile it
# from source is a twenty-minute build that usually fails instead.
FROM python:3.12-slim

# PyAV bundles its own FFmpeg libraries, so this is insurance rather than a
# requirement -- but the browser sends webm/opus, and a decode failure in
# production reads as "the microphone is broken" with nothing in the log.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Hugging Face Spaces and most managed hosts run the container as a non-root
# user. Creating one here means the app owns the directories it writes caches
# into, rather than discovering it cannot at the first request.
RUN useradd -m -u 1000 setu \
    && mkdir -p /app/data/llm_cache /app/data/voice_cache \
    && chown -R setu:setu /app

USER setu
ENV HOME=/home/setu \
    HF_HOME=/home/setu/.cache/huggingface

# Bake the ASR model into the image. Left to runtime this is a 464MB download
# that the first caller waits through -- and on any host that scales to zero,
# every caller after an idle period waits through it again.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

COPY --chown=setu:setu . .

# small, not medium: measured at 4.9s against 14.4s on one Hindi sentence, with
# extraction recovering every field from the messier transcript. See setu/voice.py.
ENV SETU_WHISPER_SIZE=small \
    PORT=7860

EXPOSE 7860

# GEMINI_API_KEY is supplied by the host as a secret, never baked in.
CMD ["sh", "-c", "uvicorn setu.api:app --host 0.0.0.0 --port ${PORT:-7860}"]
