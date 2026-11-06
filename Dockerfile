FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 RAG_INDEX_DIR=/data/index
COPY pyproject.toml README.md ./
COPY src ./src
# Add ".[embeddings]" for semantic embeddings (larger image, downloads a model on first run)
RUN pip install --no-cache-dir .
COPY data ./data
VOLUME /data
EXPOSE 8000
CMD ["rag", "serve", "--host", "0.0.0.0", "--port", "8000"]
