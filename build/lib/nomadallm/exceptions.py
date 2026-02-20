"""
NomadaLLM Exceptions

Security: All exceptions are designed to not leak sensitive information.
Compliance: Error messages are generic to end-users, detailed internally.
"""


class NomadaLLMError(Exception):
    """Base exception for all NomadaLLM errors.
    
    Security: Never exposes internal details to end-users.
    """
    
    def __init__(self, message: str, internal_details: str = None):
        self.message = message
        self.internal_details = internal_details
        super().__init__(message)
    
    def __str__(self):
        return self.message
    
    def get_internal_details(self) -> str:
        """For internal logging only - never expose to end-users."""
        return self.internal_details or self.message


class PrivacyViolationError(NomadaLLMError):
    """Raised when a privacy policy is violated.
    
    Security: Triggered when PII is detected in non-compliant contexts.
    Compliance: Required for GDPR, HIPAA, PCI-DSS.
    """
    
    def __init__(self, violation_type: str, field: str = None):
        message = "Privacy policy violation detected"
        internal = f"Violation: {violation_type}, Field: {field}"
        super().__init__(message, internal)
        self.violation_type = violation_type
        self.field = field


class ProviderError(NomadaLLMError):
    """Raised when an LLM provider fails.
    
    Security: Does not expose provider-specific error details.
    """
    
    def __init__(self, provider: str, original_error: Exception = None):
        message = "LLM provider temporarily unavailable"
        internal = f"Provider: {provider}, Error: {str(original_error)}"
        super().__init__(message, internal)
        self.provider = provider
        self.original_error = original_error


class RateLimitError(NomadaLLMError):
    """Raised when rate limits are exceeded.
    
    Security: Prevents abuse and DoS attacks.
    """
    
    def __init__(self, retry_after: int = None):
        message = "Rate limit exceeded. Please try again later."
        internal = f"Retry after: {retry_after} seconds"
        super().__init__(message, internal)
        self.retry_after = retry_after


class AuthenticationError(NomadaLLMError):
    """Raised when authentication fails.
    
    Security: Generic message to prevent credential enumeration.
    """
    
    def __init__(self, reason: str = None):
        message = "Authentication failed"
        internal = f"Reason: {reason}"
        super().__init__(message, internal)


class ConfigurationError(NomadaLLMError):
    """Raised when configuration is invalid.
    
    Security: Does not expose configuration details.
    """
    
    def __init__(self, config_key: str = None):
        message = "Invalid configuration"
        internal = f"Config key: {config_key}"
        super().__init__(message, internal)


class EncryptionError(NomadaLLMError):
    """Raised when encryption/decryption fails.
    
    Security: Critical - indicates potential security breach.
    Compliance: Must be logged and audited.
    """
    
    def __init__(self, operation: str = None):
        message = "Security operation failed"
        internal = f"Encryption operation: {operation}"
        super().__init__(message, internal)


class LicenseError(NomadaLLMError):
    """Raised when license validation fails or limits are exceeded.
    
    Security: Enforces usage limits and license compliance.
    """
    
    def __init__(self, reason: str = None, upgrade_url: str = None):
        message = "License limit reached. Upgrade at https://nomadallm.nomadahealth.com/pricing"
        internal = f"Reason: {reason}"
        super().__init__(message, internal)
        self.upgrade_url = upgrade_url or "https://nomadallm.nomadahealth.com/pricing"


class UsageLimitError(LicenseError):
    """Raised when daily usage limit is exceeded.
    
    Security: Prevents abuse and enforces fair usage.
    """
    
    def __init__(self, calls_today: int, daily_limit: int, tier: str):
        reason = f"Daily limit exceeded: {calls_today}/{daily_limit} calls (tier: {tier})"
        super().__init__(reason)
        self.calls_today = calls_today
        self.daily_limit = daily_limit
        self.tier = tier

