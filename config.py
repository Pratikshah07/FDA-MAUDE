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

# Email Configuration for Password Recovery
MAIL_SERVER = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT = int(os.getenv("MAIL_PORT", "587"))
MAIL_USE_TLS = os.getenv("MAIL_USE_TLS", "True").lower() == "true"
MAIL_USE_SSL = os.getenv("MAIL_USE_SSL", "False").lower() == "true"
MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")
MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER", MAIL_USERNAME)

# Password Reset Token Expiry (in seconds)
PASSWORD_RESET_EXPIRY = 3600  # 1 hour

# Test accounts for development/testing
# Set TEST_ACCOUNTS_ENABLED to 'False' in production to hide and avoid auto-creation
TEST_ACCOUNTS_ENABLED = os.getenv("TEST_ACCOUNTS_ENABLED", "True").lower() == "true"
TEST_USER_EMAIL = os.getenv("TEST_USER_EMAIL", "test@maude.local")
TEST_USER_PASSWORD = os.getenv("TEST_USER_PASSWORD", "Test12345")
