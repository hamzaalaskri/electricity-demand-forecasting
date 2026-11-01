FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_PYTHON_REFRESH=quiet \
    MLFLOW_DISABLE_AGENT_HINT=1

# LightGBM and XGBoost need the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# API dependencies plus exactly the libraries the exported model declares.
COPY requirements.txt .
COPY models/electricity_demand_forecaster/requirements.txt model-requirements.txt
RUN pip install -r requirements.txt -r model-requirements.txt

# Only what inference needs: code, the exported model, and the latest data.
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY data/PJME_hourly.csv data/weather_hourly.csv ./data/

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/ready', timeout=4)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
