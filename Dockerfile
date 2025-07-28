# Use official Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy your local code and model files into the container
# This assumes your local models are in a 'models' folder, etc.
COPY requirements.txt ./
COPY miniLM_NLI.py ./
COPY models/ /models/
COPY doclayout_yolo/ ./doclayout_yolo/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------------------------
# ✅ **CHANGE:** Pre-download the easyocr model during the build
# This "bakes" the model into the image so no download is needed
# when the container runs.
# --------------------------------------------------------------------
RUN python -c "import easyocr; reader = easyocr.Reader(['en'], gpu=False)"

# Install system dependencies needed for image processing and PDFs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Command to run your application
CMD ["python", "miniLM_NLI.py", "/app/input"]