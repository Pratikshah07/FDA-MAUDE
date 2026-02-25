"""
Gunicorn configuration file.
This file is read automatically by gunicorn at startup (default config path).
Settings here are overridden by explicit CLI flags, so this acts as a safe baseline.
"""
import os

# Worker configuration
# gthread: single process, multiple threads — health check is never blocked
# by a long-running pipeline job because threads handle requests concurrently.
workers = 1
worker_class = "gthread"
threads = 4

# Timeouts
timeout = 600
graceful_timeout = 120
keepalive = 5

# Logging
loglevel = "info"
accesslog = "-"   # stdout
errorlog = "-"    # stderr

# Performance: do NOT preload the app
# Preloading forces all heavy imports to happen before the port is bound,
# which causes Render's health checker to time out on cold starts.
preload_app = False

# Bind address (Render passes PORT via environment)
bind = "0.0.0.0:{}".format(os.environ.get("PORT", "8000"))
