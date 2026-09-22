FROM python:3.11-slim

# ffmpeg: mp3/m4a decoding via librosa's audioread backend
# libsndfile1: wav/flac/ogg decoding via soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV HOST=0.0.0.0 PORT=6422
EXPOSE 6422

CMD ["sh", "-c", "uvicorn main:app --host ${HOST} --port ${PORT}"]
