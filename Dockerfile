# path: Dockerfile

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY cli.py ./  # if we expose CLI

# For future FastAPI
# COPY app.py .

CMD ["python", "-m", "src.juniorclimbs.cli"]
