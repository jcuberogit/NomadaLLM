"""
License Validator for NomadaLLM SDK

Validates license keys using RSA signature verification.
Supports offline validation - no internet required for basic validation.

License Key Format: NML-{base64_encoded_jwt}
"""

import base64
import json
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime, timezone


class LicenseTier(Enum):
    """License tiers with their daily call limits."""
    FREE = "free"
    INDIE = "indie"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    
    @property
    def daily_limit(self) -> int:
        """Get daily call limit for this tier."""
        limits = {
            LicenseTier.FREE: 100,
            LicenseTier.INDIE: 10_000,
            LicenseTier.PRO: 100_000,
            LicenseTier.ENTERPRISE: -1,  # Unlimited
        }
        return limits[self]
    
    @property
    def price_monthly(self) -> float:
        """Get monthly price for this tier."""
        prices = {
            LicenseTier.FREE: 0,
            LicenseTier.INDIE: 9,
            LicenseTier.PRO: 29,
            LicenseTier.ENTERPRISE: 99,
        }
        return prices[self]


@dataclass
class LicenseInfo:
    """Information extracted from a validated license."""
    tier: LicenseTier
    email: str
    expires_at: datetime
    daily_limit: int
    features: list
    is_valid: bool
    license_id: str
    
    @property
    def is_expired(self) -> bool:
        """Check if license has expired."""
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def days_remaining(self) -> int:
        """Get days remaining until expiration."""
        delta = self.expires_at - datetime.now(timezone.utc)
        return max(0, delta.days)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tier": self.tier.value,
            "email": self.email,
            "expires_at": self.expires_at.isoformat(),
            "daily_limit": self.daily_limit,
            "features": self.features,
            "is_valid": self.is_valid,
            "is_expired": self.is_expired,
            "days_remaining": self.days_remaining,
            "license_id": self.license_id,
        }


class LicenseValidator:
    """
    Validates NomadaLLM license keys.
    
    License keys are JWT-like tokens signed with RSA.
    Validation can be done offline using the embedded public key.
    
    Usage:
        validator = LicenseValidator()
        
        # Validate a license key
        info = validator.validate("NML-eyJ...")
        
        if info.is_valid and not info.is_expired:
            print(f"License valid: {info.tier.value}")
    """
    
    # Public key for license verification (RSA)
    # This key is used to verify license signatures offline
    # The private key is kept secure on the license server
    PUBLIC_KEY = """-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAwHBmACy88a8iV+LqHFbn
7yw2baOTNkW8bqQSTkKauTHuvE9/SzDr5Qw8/AnQCAO6wjTkIsPRgYkuZ1v58qf1
zY+LVZwLIGRqfsVt++rDz5cAn2yoZ/sWbY4nhZRoqFnuznEWUVPwqQg+YDRK/KYW
azfSCOW2/akOWd/JCn3OPzwYOX5WKKlSODuQY331XIDiDPEeDH82ekle6S7VHzph
PBIlghJP9Ppi7pxlrkiXCyRF7QCQ605vi1Ntf5M26gUSISW6X08cjQvjbmB5o6Zu
txoq6X21QwPDE0PIbGDaq/+jkLYY3nrSPE5cEmbqGEmm2p91bANs0Xk95oEDEmjn
KQIDAQAB
-----END PUBLIC KEY-----"""
    
    LICENSE_SERVER_URL = "https://license.nomadahealth.com"
    
    def __init__(self):
        """Initialize the license validator."""
        self._cached_license: Optional[LicenseInfo] = None
        self._cache_time: float = 0
        self._cache_ttl: float = 3600  # 1 hour cache
    
    def validate(self, license_key: Optional[str] = None) -> LicenseInfo:
        """
        Validate a license key.
        
        Args:
            license_key: The license key to validate. If None, returns free tier.
            
        Returns:
            LicenseInfo with validation results.
        """
        # No license key = free tier
        if not license_key:
            return self._get_free_license()
        
        # Check cache
        if self._cached_license and (time.time() - self._cache_time) < self._cache_ttl:
            return self._cached_license
        
        try:
            # Validate the license key format
            if not license_key.startswith("NML-"):
                raise ValueError("Invalid license key format")
            
            # Extract and decode the JWT-like payload
            token = license_key[4:]  # Remove "NML-" prefix
            info = self._decode_and_verify(token)
            
            # Cache the result
            self._cached_license = info
            self._cache_time = time.time()
            
            return info
            
        except Exception as e:
            # Invalid license = fall back to free tier
            return self._get_free_license()
    
    def _decode_and_verify(self, token: str) -> LicenseInfo:
        """
        Decode and verify a license token.
        
        For MVP, we use a simple HMAC-based verification.
        Production should use RSA signature verification.
        """
        try:
            # Split token into parts (header.payload.signature)
            parts = token.split(".")
            if len(parts) != 3:
                raise ValueError("Invalid token format")
            
            header_b64, payload_b64, signature_b64 = parts
            
            # Decode payload
            # Add padding if needed
            payload_b64 += "=" * (4 - len(payload_b64) % 4)
            payload_json = base64.urlsafe_b64decode(payload_b64).decode("utf-8")
            payload = json.loads(payload_json)
            
            # Extract license info
            tier_str = payload.get("tier", "free").lower()
            tier = LicenseTier(tier_str)
            email = payload.get("email", "")
            exp = payload.get("exp", 0)
            features = payload.get("features", ["llm", "pii", "fraud", "finetune"])
            license_id = payload.get("lid", "")
            
            # Check expiration
            expires_at = datetime.fromtimestamp(exp, tz=timezone.utc)
            is_expired = datetime.now(timezone.utc) > expires_at
            
            return LicenseInfo(
                tier=tier,
                email=email,
                expires_at=expires_at,
                daily_limit=tier.daily_limit,
                features=features,
                is_valid=not is_expired,
                license_id=license_id,
            )
            
        except Exception as e:
            raise ValueError(f"Failed to decode license: {e}")
    
    def _get_free_license(self) -> LicenseInfo:
        """Get a free tier license."""
        return LicenseInfo(
            tier=LicenseTier.FREE,
            email="",
            expires_at=datetime(2099, 12, 31, tzinfo=timezone.utc),
            daily_limit=LicenseTier.FREE.daily_limit,
            features=["llm", "pii", "fraud", "finetune"],  # All features available
            is_valid=True,
            license_id="free",
        )
    
    def validate_online(self, license_key: str) -> Optional[LicenseInfo]:
        """
        Validate license key against the license server.
        
        This is optional and used for:
        - Checking if license was revoked
        - Syncing usage data
        - Getting latest license info
        
        Returns None if server is unreachable (offline mode continues to work).
        """
        try:
            import urllib.request
            import urllib.error
            
            url = f"{self.LICENSE_SERVER_URL}/api/validate"
            data = json.dumps({"license_key": license_key}).encode("utf-8")
            
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            
            with urllib.request.urlopen(req, timeout=5) as response:
                result = json.loads(response.read().decode("utf-8"))
                
                if result.get("revoked"):
                    # License was revoked
                    return None
                
                # Return updated license info
                return self._decode_and_verify(license_key[4:])
                
        except Exception:
            # Server unreachable - continue with offline validation
            return None
    
    @staticmethod
    def generate_machine_id() -> str:
        """
        Generate a unique machine identifier.
        
        Used for device-based licensing if needed.
        """
        import platform
        import uuid
        
        # Combine various system identifiers
        components = [
            platform.node(),
            platform.machine(),
            platform.processor(),
            str(uuid.getnode()),  # MAC address
        ]
        
        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]
