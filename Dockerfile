FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

ARG GIT_REMOTE_URL
RUN git init && git lfs install && git remote add origin "$GIT_REMOTE_URL" && git lfs pull

ENV HF_HOME=/app/.cache

RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONPATH=/app
ENV DB_PATH=/app/database.db

# Pre-download embedding model to /app/.cache (persists in final image)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

ENV PYTHONPATH=/app
ENV DB_PATH=/app/database.db
ENV CHROMA_PATH=/app/chroma_store
ENV HF_HOME=/app/.cache

EXPOSE 7860

CMD ["python", "app/startup.py"]
