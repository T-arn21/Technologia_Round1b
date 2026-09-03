FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache EasyOCR model weights into /root/.EasyOCR
RUN python -c "import easyocr; reader = easyocr.Reader(['en'], gpu=False)"

COPY models/ /models/
COPY doclayout_yolo/ ./doclayout_yolo/
COPY miniLM_NLI.py ./

RUN mkdir -p /app/input /app/output

CMD ["python", "miniLM_NLI.py", "/app/input"]

