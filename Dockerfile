# syntax = docker/dockerfile:1.4

# Plain Python rather than ghcr.io/leolani/cltl-base: the knowledge graph and
# LLM dependencies (cltl.brain, openai, pydantic) are not in the platform's
# offline package registry, so everything is installed from PyPI and GitHub.
# Python 3.10, because emissor only installs cleanly under it.
FROM python:3.10-slim

LABEL org.opencontainers.image.source="https://github.com/leolani/cltl-custom-diabetes"
LABEL org.opencontainers.image.description="Knowledge graph driven diabetes lifestyle coach for a Leolani deployment"
LABEL org.opencontainers.image.licenses="MIT"

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /cltl-custom-diabetes

COPY setup.py requirements.txt README.md VERSION ./
COPY src ./src
RUN pip install --no-cache-dir -r requirements.txt

COPY intents ./intents
COPY config ./config

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8008/health || exit 1

# The working directory matters: the kg-chat code finds src/cltl and intents/
# relative to it.
CMD ["python", "src/main.py"]
