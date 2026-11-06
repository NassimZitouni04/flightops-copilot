FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir .

# The app finds its data relative to the source tree, so import it from there.
ENV PYTHONPATH=/app/src
ENV FASTEMBED_CACHE_PATH=/app/.cache/fastembed

# Download the search models and embed the corpus now, so containers start ready and work offline.
RUN python -c "from flightops.retrieval import Retriever; Retriever().search('altitude limit', k=1)"

# Run as a non-root user that can only write its own cache and traces.
RUN useradd --create-home appuser && mkdir -p /app/traces && chown -R appuser:appuser /app/.cache /app/traces
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"]

# Pass the API key at run time, for example with --env-file. It is never stored in the image.
CMD ["uvicorn", "flightops.api:app", "--host", "0.0.0.0", "--port", "8000"]
