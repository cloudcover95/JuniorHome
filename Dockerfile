# path: Dockerfile

FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/juniorclimbs ./src/juniorclimbs
COPY cli.py .

# For future FastAPI admin UI
# EXPOSE 8000

CMD ["python", "-m", "src.juniorclimbs.cli"]
