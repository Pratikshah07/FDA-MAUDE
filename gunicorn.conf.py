"""
Gunicorn configuration file.
This file is read automatically by gunicorn at startup (default config path).
Settings here are overridden by explicit CLI flags, so this acts as a safe baseline.
"""
import os

# Worker configuration
workers = 1
worker_class = "sync"
threads = 1

# Timeouts
timeout = 600
graceful_timeout = 120
keepalive = 5

# Logging
loglevel = "info"
accesslog = "-"   # stdout
errorlog = "-"    # stderr

# Preload the app in the master process so workers fork() from already-loaded
# memory (copy-on-write). This avoids each worker independently importing all
# modules, which on Render's free tier (512 MB RAM) causes the OS OOM-killer
# to silently terminate workers before they can serve any requests.
# The /health endpoint responds instantly after fork, so Render's health check
# no longer times out.
preload_app = True

# Bind address (Render passes PORT via environment)
bind = "0.0.0.0:{}".format(os.environ.get("PORT", "8000"))
