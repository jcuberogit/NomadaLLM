"""
License Generator for NomadaLLM SDK

Generates JWT-based license keys signed with RSA.
Used after PayPal payment to create license for customer.
"""

import base64
import json
import hashlib
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend


class LicenseGenerator:
    """
    Generates NomadaLLM license keys.
    
    License Format: NML-{base64_header}.{base64_payload}.{base64_signature}
    
    The license is a JWT-like token that can be validated offline
    using the public key embedded in the SDK.
    """
    
    def __init__(self, private_key_pem: Optional[str] = None):
        """
        Initialize the license generator.
        
        Args:
            private_key_pem: RSA private key in PEM format.
                            If None, generates a new key pair.
        """
        if private_key_pem:
            self.private_key = serialization.load_pem_private_key(
                private_key_pem.encode(),
                password=None,
                backend=default_backend()
            )
        else:
            # Generate new key pair (only for initial setup)
            self.private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )
        
        self.public_key = self.private_key.public_key()
    
    def generate_license(
        self,
        email: str,
        tier: str,
        months: int = 1,
        paypal_order_id: Optional[str] = None
    ) -> dict:
        """
        Generate a new license key.
        
        Args:
            email: Customer email
            tier: License tier (free, indie, pro, enterprise)
            months: License duration in months
            paypal_order_id: PayPal order ID for reference
            
        Returns:
            dict with license_key, api_key, expires_at, etc.
        """
        # Generate unique license ID
        license_id = secrets.token_hex(8)
        
        # Calculate expiration
        expires_at = datetime.now(timezone.utc) + timedelta(days=30 * months)
        
        # Create payload
        payload = {
            "lid": license_id,
            "email": email,
            "tier": tier,
            "exp": int(expires_at.timestamp()),
            "iat": int(datetime.now(timezone.utc).timestamp()),
            "features": ["llm", "pii", "fraud", "finetune"],  # All features
            "paypal": paypal_order_id,
        }
        
        # Create header
        header = {
            "alg": "RS256",
            "typ": "NML"
        }
        
        # Encode header and payload
        header_b64 = self._base64url_encode(json.dumps(header))
        payload_b64 = self._base64url_encode(json.dumps(payload))
        
        # Create signature
        message = f"{header_b64}.{payload_b64}".encode()
        signature = self.private_key.sign(
            message,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
        signature_b64 = self._base64url_encode(signature)
        
        # Create license key
        license_key = f"NML-{header_b64}.{payload_b64}.{signature_b64}"
        
        # Create simple API key for quick validation
        api_key = f"nllm_{tier}_{license_id}"
        
        return {
            "license_key": license_key,
            "api_key": api_key,
            "license_id": license_id,
            "email": email,
            "tier": tier,
            "expires_at": expires_at.isoformat(),
            "paypal_order_id": paypal_order_id,
        }
    
    def get_public_key_pem(self) -> str:
        """Get the public key in PEM format for embedding in SDK."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()
    
    def get_private_key_pem(self) -> str:
        """Get the private key in PEM format (KEEP SECRET!)."""
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode()
    
    @staticmethod
    def _base64url_encode(data) -> str:
        """Base64 URL-safe encoding without padding."""
        if isinstance(data, str):
            data = data.encode()
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


# Singleton instance with generated keys
_generator: Optional[LicenseGenerator] = None


def get_generator() -> LicenseGenerator:
    """Get or create the license generator singleton."""
    global _generator
    if _generator is None:
        # In production, load from environment variable or secure storage
        import os
        private_key = os.getenv("NOMADALLM_LICENSE_PRIVATE_KEY")
        _generator = LicenseGenerator(private_key)
    return _generator


def generate_license(
    email: str,
    tier: str,
    months: int = 1,
    paypal_order_id: Optional[str] = None
) -> dict:
    """
    Convenience function to generate a license.
    
    Usage:
        from nomadallm.licensing.generator import generate_license
        
        result = generate_license(
            email="customer@example.com",
            tier="pro",
            months=12,
            paypal_order_id="ABC123"
        )
        
        print(result["license_key"])  # NML-eyJ...
        print(result["api_key"])      # nllm_pro_abc123
    """
    return get_generator().generate_license(email, tier, months, paypal_order_id)
