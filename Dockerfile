# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e
ARG ACCELERATOR=cpu

FROM ghcr.io/astral-sh/uv:0.11.21@sha256:ff07b86af50d4d9391d9daf4ff89ce427bc544f9aae87057e69a1cc0aa369946 AS uv

FROM python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c AS builder

ARG ACCELERATOR
ARG TARGETARCH
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app
COPY --from=uv /uv /usr/local/bin/uv
COPY pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./
RUN case "${ACCELERATOR}:${TARGETARCH}" in \
      cpu:amd64|cpu:arm64|cu124:amd64) ;; \
      *) echo "unsupported accelerator/platform: ${ACCELERATOR}/${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && uv sync --frozen --no-dev --no-install-project --extra "${ACCELERATOR}"
COPY src/ src/
RUN uv sync --frozen --no-dev --no-editable --extra "${ACCELERATOR}"

FROM python:3.12.12-slim@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c

ARG ACCELERATOR
LABEL org.opencontainers.image.source="https://github.com/neurwerk/k8s_stack_pii_engine"
LABEL org.opencontainers.image.description="Neurwerk PII policy engine with offline spaCy baseline models"
LABEL org.opencontainers.image.accelerator="${ACCELERATOR}"

ENV PATH=/app/.venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    PII_ENGINE_IMAGE_ACCELERATOR=${ACCELERATOR} \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system --gid 1000 app \
    && useradd --system --uid 1000 --gid app --home-dir /app appuser

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/LICENSE /licenses/neurwerk-pii-engine/LICENSE
COPY --from=builder /app/THIRD_PARTY_NOTICES.md /licenses/neurwerk-pii-engine/THIRD_PARTY_NOTICES.md

USER appuser
EXPOSE 8443 8001
CMD ["pii-engine"]
