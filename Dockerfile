FROM python:3.12-slim

# ffmpeg is required by yt-dlp (merging/converting streams) and by the
# compression step. curl/ca-certificates keep pip + yt-dlp's own network
# calls working reliably.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Where downloaded files land inside the container. Mount a volume here if
# you want files to survive a restart; otherwise they're ephemeral, which
# is fine since users are meant to grab them immediately.
RUN mkdir -p /app/downloads

ENV PORT=8080
EXPOSE 8080

# --workers: number of OS processes (each has its own JOBS dict, so keep at
# 1 unless you also move job state to something shared like Redis).
# --threads: handled inside the app via ThreadPoolExecutor already.
# --timeout 0: downloads can run long; don't let gunicorn kill the worker.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 8 --timeout 0 app:app
