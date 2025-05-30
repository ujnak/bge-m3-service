FROM python:3.12
WORKDIR /app

COPY embed_service.py .
RUN pip install torch transformers fastapi uvicorn FlagEmbedding

CMD ["uvicorn", "embed_service:app", "--host", "0.0.0.0", "--port", "7999", "--log-level", "info", "--access-log"]

