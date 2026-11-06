FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir .

# The app looks for data relative to the source tree, so run it from there.
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn", "flightops.api:app", "--host", "0.0.0.0", "--port", "8000"]
