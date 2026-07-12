FROM python:3.11-slim

# ติดตั้ง ffmpeg (จำเป็นสำหรับ decode m4a → PCM)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# copy requirements ก่อน (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy โค้ดทั้งหมด
COPY . .

# Render กำหนด PORT ผ่าน env
ENV PORT=10000
EXPOSE 10000

# gunicorn 1 worker (ประหยัด RAM) + timeout สูง
CMD gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120 --workers 1 --max-requests 50 --max-requests-jitter 10
