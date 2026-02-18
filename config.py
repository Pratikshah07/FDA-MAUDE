"""
Configuration file for MAUDE processing pipeline.
"""
import os

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables only

# Groq API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # Using Groq's latest model

# Schema Definition (Phase 0 - Schema Freeze)
COLUMNS_TO_DELETE = [
    "Report Number",
    "PMA/PMN",
    "PMA/PMN Number",  # Also handle variations - any header starting with "pma/pmn" (normalized)
    "PMN Number",
    "Web Address",
    "Product Code",
    "Exemption Number",
    "Number of Events"
]


COLUMNS_TO_MODIFY = [
    "Event Date",
    "Date Received",
    "Manufacturer Name",  # or equivalent column names
    "Device Problem"
]

COLUMNS_TO_ADD = [
    "IMDRF Code"
]

# Legal suffixes to remove from manufacturer names
LEGAL_SUFFIXES = [
    "ltd", "llp", "inc", "corp", "company", "co", "gmbh", "ag", "sa", "sarl",
    "bv", "plc", "pvt", "limited"
]

# Date format for output
DATE_FORMAT = "%d-%m-%Y"

# IMDRF Annexure structure (will be loaded from file or defined here)
# This is a placeholder - actual structure should come from Annexure A-G files
IMDRF_STRUCTURE = {
    # Level 1: High-level categories
    # Level 2: Mid-level categories  
    # Level 3: Specific codes
    # Structure: {level: {code: {description: str, children: dict}}}
}

# Flask Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "maude-secret-key-change-in-production-2024")

# Firebase Configuration (Frontend JS SDK)
FIREBASE_CONFIG = {
    "apiKey": os.getenv("FIREBASE_API_KEY", ""),
    "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
    "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
    "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
    "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
    "appId": os.getenv("FIREBASE_APP_ID", ""),
}

# Firebase Admin SDK (Backend)
# Try to use Firebase service account path from env, or create temp file from JSON content
import json
import tempfile

_firebase_service_account_path = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "")
if not _firebase_service_account_path or not os.path.exists(_firebase_service_account_path):
    # Try to load from JSON content in environment variable
    _firebase_json_content = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
    if _firebase_json_content:
        try:
            # Create a temporary file with the JSON content
            _temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json')
            json.dump(json.loads(_firebase_json_content), _temp_file)
            _temp_file.close()
            FIREBASE_SERVICE_ACCOUNT_PATH = _temp_file.name
        except Exception:
            # Fallback to default location
            FIREBASE_SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firebase-service-account.json")
    else:
        FIREBASE_SERVICE_ACCOUNT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "firebase-service-account.json")
else:
    FIREBASE_SERVICE_ACCOUNT_PATH = _firebase_service_account_path
