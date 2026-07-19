FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt requirements-transform.txt ./
RUN pip install --no-cache-dir -r requirements-transform.txt

COPY . .

# Runs one ingestion snapshot by default. Override the command for other stages, e.g.:
#   docker run --rm -v "$PWD/data:/app/data" cincy-transit python ingestion/fetch_realtime.py
#   docker run --rm -e MOTHERDUCK_TOKEN=... cincy-transit sh -c "cd transform && dbt build --profiles-dir ."
#   docker run --rm -e MOTHERDUCK_TOKEN=... cincy-transit python ml/train.py
CMD ["python", "ingestion/fetch_realtime.py"]
