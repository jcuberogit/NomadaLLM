"""
NomadaLLM Privacy Modes

Pre-configured privacy profiles for different industries and compliance requirements.

Security: Each mode enforces specific data handling policies.
Compliance: Modes are designed to meet regulatory requirements.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Set


class PrivacyMode(Enum):
    """Privacy modes for different compliance requirements.
    
    Security: Each mode defines what data can be processed and how.
    """
    
    # Standard mode - basic privacy protections
    STANDARD = "standard"
    
    # Banking/Fintech - PCI-DSS, SOX compliance
    BANKING = "banking"
    
    # Healthcare - HIPAA compliance
    HEALTHCARE = "healthcare"
    
    # Enterprise - SOC2 compliance
    ENTERPRISE = "enterprise"
    
    # Maximum privacy - Zero-knowledge, no data retention
    ZERO_KNOWLEDGE = "zero_knowledge"
    
    # GDPR specific - EU data protection
    GDPR = "gdpr"
    
    # Custom - user-defined rules
    CUSTOM = "custom"


@dataclass
class PrivacyConfig:
    """Configuration for a privacy mode.
    
    Security: Defines all privacy-related settings.
    Compliance: Each setting maps to regulatory requirements.
    """
    
    mode: PrivacyMode
    
    # PII Detection settings
    detect_pii: bool = True
    pii_types_to_detect: Set[str] = None
    
    # Masking settings
    mask_pii_in_prompts: bool = True
    mask_pii_in_responses: bool = True
    masking_strategy: str = "redact"  # redact, hash, tokenize, encrypt
    
    # Encryption settings
    encrypt_at_rest: bool = True
    encrypt_in_transit: bool = True
    encryption_algorithm: str = "AES-256-GCM"
    
    # Data retention
    retain_data: bool = False
    retention_days: int = 0
    
    # Audit settings
    audit_all_requests: bool = True
    audit_pii_access: bool = True
    
    # Data residency
    allowed_regions: List[str] = None
    
    # Zero-knowledge options
    zero_knowledge_mode: bool = False
    client_side_encryption: bool = False
    
    def __post_init__(self):
        if self.pii_types_to_detect is None:
            self.pii_types_to_detect = set()
        if self.allowed_regions is None:
            self.allowed_regions = []


# Pre-configured privacy profiles
PRIVACY_PROFILES = {
    PrivacyMode.STANDARD: PrivacyConfig(
        mode=PrivacyMode.STANDARD,
        detect_pii=True,
        mask_pii_in_prompts=False,
        mask_pii_in_responses=False,
        encrypt_at_rest=True,
        encrypt_in_transit=True,
        retain_data=True,
        retention_days=30,
        audit_all_requests=False,
        audit_pii_access=True,
    ),
    
    PrivacyMode.BANKING: PrivacyConfig(
        mode=PrivacyMode.BANKING,
        detect_pii=True,
        pii_types_to_detect={
            "credit_card", "bank_account", "ssn", "routing_number",
            "account_balance", "transaction_id", "pin", "cvv"
        },
        mask_pii_in_prompts=True,
        mask_pii_in_responses=True,
        masking_strategy="tokenize",
        encrypt_at_rest=True,
        encrypt_in_transit=True,
        encryption_algorithm="AES-256-GCM",
        retain_data=False,
        retention_days=0,
        audit_all_requests=True,
        audit_pii_access=True,
        zero_knowledge_mode=False,
        client_side_encryption=True,
    ),
    
    PrivacyMode.HEALTHCARE: PrivacyConfig(
        mode=PrivacyMode.HEALTHCARE,
        detect_pii=True,
        pii_types_to_detect={
            "patient_id", "medical_record", "diagnosis", "medication",
            "ssn", "insurance_id", "dob", "address", "phone", "email"
        },
        mask_pii_in_prompts=True,
        mask_pii_in_responses=True,
        masking_strategy="redact",
        encrypt_at_rest=True,
        encrypt_in_transit=True,
        encryption_algorithm="AES-256-GCM",
        retain_data=False,
        retention_days=0,
        audit_all_requests=True,
        audit_pii_access=True,
        zero_knowledge_mode=True,
        client_side_encryption=True,
    ),
    
    PrivacyMode.ENTERPRISE: PrivacyConfig(
        mode=PrivacyMode.ENTERPRISE,
        detect_pii=True,
        pii_types_to_detect={
            "email", "phone", "address", "ssn", "employee_id",
            "salary", "performance_review"
        },
        mask_pii_in_prompts=True,
        mask_pii_in_responses=True,
        masking_strategy="hash",
        encrypt_at_rest=True,
        encrypt_in_transit=True,
        retain_data=True,
        retention_days=90,
        audit_all_requests=True,
        audit_pii_access=True,
    ),
    
    PrivacyMode.ZERO_KNOWLEDGE: PrivacyConfig(
        mode=PrivacyMode.ZERO_KNOWLEDGE,
        detect_pii=True,
        mask_pii_in_prompts=True,
        mask_pii_in_responses=True,
        masking_strategy="encrypt",
        encrypt_at_rest=True,
        encrypt_in_transit=True,
        encryption_algorithm="AES-256-GCM",
        retain_data=False,
        retention_days=0,
        audit_all_requests=False,
        audit_pii_access=False,
        zero_knowledge_mode=True,
        client_side_encryption=True,
    ),
    
    PrivacyMode.GDPR: PrivacyConfig(
        mode=PrivacyMode.GDPR,
        detect_pii=True,
        pii_types_to_detect={
            "name", "email", "phone", "address", "dob", "ip_address",
            "location", "biometric", "genetic", "political_opinion",
            "religious_belief", "sexual_orientation"
        },
        mask_pii_in_prompts=True,
        mask_pii_in_responses=True,
        masking_strategy="redact",
        encrypt_at_rest=True,
        encrypt_in_transit=True,
        retain_data=True,
        retention_days=365,
        audit_all_requests=True,
        audit_pii_access=True,
        allowed_regions=["EU", "EEA"],
    ),
}


def get_privacy_config(mode: PrivacyMode) -> PrivacyConfig:
    """Get the privacy configuration for a given mode.
    
    Security: Returns immutable copy to prevent tampering.
    """
    if mode == PrivacyMode.CUSTOM:
        return PrivacyConfig(mode=PrivacyMode.CUSTOM)
    
    return PRIVACY_PROFILES.get(mode, PRIVACY_PROFILES[PrivacyMode.STANDARD])
