FROM python:3.14-slim

ENV HF_HOME=/opt/huggingface
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

RUN mkdir -p /opt/huggingface /app/data/uploads \
    && python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app /opt/huggingface

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT', '8000') + '/health', timeout=3)"

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
