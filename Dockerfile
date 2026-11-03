FROM python:3.12-slim

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY src ./src
COPY models/churn_model.pkl models/preprocessor.pkl models/metrics.json ./models/

# Run as non-root. data/ holds the prediction log: mount a volume there to keep it
RUN useradd --create-home appuser && mkdir data && chown appuser data
USER appuser

# numba, pulled in by shap, needs a writable cache directory for a non-root user
ENV NUMBA_CACHE_DIR=/tmp/numba

EXPOSE 8000
HEALTHCHECK --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
