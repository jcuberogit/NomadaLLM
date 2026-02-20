"""
NomadaLLM - Universal LLM SDK with Privacy-First Architecture

A universal SDK that provides native AI/LLM integration with built-in
data privacy controls for sensitive industries (banking, healthcare, enterprise).

Usage:
    from nomadallm import NomadaLLM
    
    # Free tier (100 calls/day, all features)
    llm = NomadaLLM(privacy_mode="banking")
    response = llm.chat("Analyze this statement", user_data=data)
    
    # With license key (higher limits)
    llm = NomadaLLM(license_key="NML-xxx", privacy_mode="healthcare")

Security: All data is encrypted in transit (TLS 1.3) and at rest (AES-256).
Compliance: SOC2, GDPR, HIPAA, PCI-DSS ready.
"""

__version__ = "0.1.0"
__author__ = "Nomada Health"
__license__ = "Proprietary"

from nomadallm.client import NomadaLLM
from nomadallm.privacy import PrivacyLayer, PrivacyMode
from nomadallm.providers import Provider, ProviderConfig, EmbeddedProvider
from nomadallm.licensing import LicenseValidator, LicenseInfo, LicenseTier, UsageTracker
from nomadallm.finetune import FineTuner, DatasetLoader, DatasetFormat
from nomadallm.rag import RAGProvider, RAGContext, RAGResult
from nomadallm.exceptions import (
    NomadaLLMError,
    PrivacyViolationError,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    LicenseError,
    UsageLimitError,
)

__all__ = [
    "NomadaLLM",
    "PrivacyLayer",
    "PrivacyMode",
    "Provider",
    "ProviderConfig",
    "EmbeddedProvider",
    "LicenseValidator",
    "LicenseInfo",
    "LicenseTier",
    "UsageTracker",
    "FineTuner",
    "DatasetLoader",
    "DatasetFormat",
    "RAGProvider",
    "RAGContext",
    "RAGResult",
    "NomadaLLMError",
    "PrivacyViolationError",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "LicenseError",
    "UsageLimitError",
]
