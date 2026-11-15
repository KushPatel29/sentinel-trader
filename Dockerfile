FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# libgomp is needed by LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dashboard]"

COPY config ./config
COPY dashboard ./dashboard

# data/, models/ and reports/ are mounted as volumes so they survive container restarts
VOLUME ["/app/data", "/app/models", "/app/reports"]

CMD ["sentinel", "live"]
