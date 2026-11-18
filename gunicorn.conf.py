"""Gunicorn configuration for Render (and any POSIX container host).

Gunicorn is intentionally limited to ONE worker process:

* the video job queue and its single background worker thread live in-process,
  so a second process would run an independent queue and never see jobs
  submitted to the first;
* SQLite is the system of record - a single process avoids multi-writer
  contention (WAL is enabled, but one writer is the safe deployment shape);
* the YOLO checkpoints are loaded lazily into memory once per process.

Threads keep the dashboard responsive while one request waits on disk/CPU.
Do NOT add ``preload_app``: the video worker thread must start inside the
serving process, not before forking.
"""

from __future__ import annotations

import os

bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"
workers = int(os.environ.get("WEB_CONCURRENCY", "1"))
threads = int(os.environ.get("SITEGUARD_GUNICORN_THREADS", "8"))

# 256 MB video uploads and long CPU-bound requests need more than the 30 s
# default. Video inference itself runs in a background thread, so this only
# bounds a single HTTP request.
timeout = int(os.environ.get("SITEGUARD_GUNICORN_TIMEOUT", "600"))
graceful_timeout = 60
keepalive = 5

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("SITEGUARD_LOG_LEVEL", "info").lower()

preload_app = False
