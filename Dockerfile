FROM python:3.11-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim AS final
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY tests/ ./tests/
RUN touch route_log.jsonl && chown -R appuser:appgroup /app
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["python", "-m", "src.main"]
