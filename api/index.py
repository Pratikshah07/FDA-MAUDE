"""
Vercel serverless function wrapper for Flask application.
"""
import sys
import os

# Add parent directory to path to import app
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app import app

# Vercel expects the app to be accessible
# The Flask app instance is automatically detected by Vercel's Python runtime
