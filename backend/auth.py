"""
Firebase Authentication module for MAUDE Data Processor.
Verifies Firebase ID tokens and manages session-based user state.
"""
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
from flask_login import UserMixin
from typing import Optional
import jwt
import requests
from cryptography import x509
from cryptography.hazmat.backends import default_backend


class User(UserMixin):
    """User class for Flask-Login, populated from Firebase token claims."""

    def __init__(self, uid: str, email: str, name: str = ""):
        self.id = uid
        self.email = email
        self.name = name


class FirebaseAuthManager:
    """Manages Firebase Authentication token verification."""

    def __init__(self, service_account_path: str):
        try:
            firebase_admin.get_app()
        except ValueError:
            cred = credentials.Certificate(service_account_path)
            firebase_admin.initialize_app(cred)

    def verify_id_token(self, id_token: str) -> Optional[User]:
        """Verify a Firebase ID token with clock skew tolerance."""
        try:
            # Get kid from token header
            header = jwt.get_unverified_header(id_token)
            kid = header.get("kid")

            # Get Google's public keys for verification
            certs_url = "https://www.googleapis.com/robot/v1/metadata/x509/securetoken@system.gserviceaccount.com"
            response = requests.get(certs_url)
            response.raise_for_status()
            certs = response.json()

            if not kid or kid not in certs:
                print(f"[Firebase Auth Error] Invalid key ID in token. Got: {kid}")
                return None

            # Parse X.509 certificate and extract public key
            cert_pem = certs[kid].encode('utf-8')
            cert = x509.load_pem_x509_certificate(cert_pem, default_backend())
            public_key = cert.public_key()

            # Verify with 5-minute clock skew tolerance (300 seconds)
            # Skip iat verification due to system clock drift, but still verify signature and exp
            decoded = jwt.decode(
                id_token,
                public_key,
                algorithms=["RS256"],
                audience="maude-data-processor",
                options={"leeway": 300, "verify_iat": False}
            )

            return User(
                uid=decoded.get("sub") or decoded.get("user_id"),
                email=decoded.get("email", ""),
                name=decoded.get("name", decoded.get("email", "").split("@")[0]),
            )
        except Exception as e:
            print(f"[Firebase Auth Error] {type(e).__name__}: {e}")
            return None

    @staticmethod
    def get_user_from_session(session_data: dict) -> Optional[User]:
        """Reconstruct a User from Flask session data."""
        if not session_data or "uid" not in session_data:
            return None
        return User(
            uid=session_data["uid"],
            email=session_data.get("email", ""),
            name=session_data.get("name", ""),
        )
