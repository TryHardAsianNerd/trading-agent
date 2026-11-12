# Background worker image (D-025). The trading agent itself still runs on the host.
FROM python:3.13-slim

WORKDIR /app
# Pinned dependency set first: rebuilds after a code-only change reuse this layer.
COPY requirements.lock requirements-sentiment.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock
# FinBERT's stack (D-026), CPU-only torch: PyPI's default Linux wheel bundles CUDA
# (gigabytes more) and there is no GPU here.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-sentiment.lock
# transformers would otherwise start a background thread on every model load that
# asks a third-party Space to convert the pinned .bin and downloads an UNMERGED
# community safetensors PR into the cache (D-026). Progress bars and hub chatter
# are plain text in a JSON log stream.
ENV DISABLE_SAFETENSORS_CONVERSION=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error \
    HF_HUB_VERBOSITY=error
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

# Celery refuses to be comfortable as root; nothing here needs it.
RUN useradd --create-home worker
USER worker
# Exists (owned by worker) before the named volume mounts here, so the volume inherits it.
RUN mkdir -p /home/worker/.cache/huggingface

# models/ is bind-mounted from the host so the agent loads what the worker trains.
CMD ["celery", "-A", "trading_agent.worker", "worker", "--beat", "--concurrency=1", \
     "--loglevel=INFO", "--schedule=/tmp/celerybeat-schedule"]
