# JUROR backend container.
# Build:  docker build -t juror-backend .
# Run:    docker run -p 8000:8000 \
#                    -e ANTHROPIC_API_KEY=sk-ant-... \
#                    juror-backend

FROM python:3.11-slim

# system deps for torch / sentencepiece
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (better Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App + models + frontend
COPY . /app/server/
# Front-end lives one directory up so the FastAPI server can mount it
COPY ../demo /app/demo/

# Pre-download the NLI model into the image (optional; saves first-run latency)
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
    name='cross-encoder/nli-deberta-v3-small'; \
    AutoTokenizer.from_pretrained(name); \
    AutoModelForSequenceClassification.from_pretrained(name)" || true

WORKDIR /app/server
EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
