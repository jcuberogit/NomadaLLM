"""
NomadaLLM PII Detector

Detects Personally Identifiable Information (PII) in text using
pattern matching and ML-based recognition.

Security: Core component for privacy compliance.
Compliance: Required for GDPR, HIPAA, PCI-DSS, SOC2.
"""

import re
import hashlib
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum


class PIIType(Enum):
    """Types of PII that can be detected.
    
    Security: Comprehensive list covering major compliance frameworks.
    """
    # Financial
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    ROUTING_NUMBER = "routing_number"
    CVV = "cvv"
    PIN = "pin"
    
    # Identity
    SSN = "ssn"
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    
    # Contact
    EMAIL = "email"
    PHONE = "phone"
    ADDRESS = "address"
    
    # Personal
    NAME = "name"
    DOB = "dob"
    AGE = "age"
    
    # Healthcare
    MEDICAL_RECORD = "medical_record"
    PATIENT_ID = "patient_id"
    DIAGNOSIS = "diagnosis"
    MEDICATION = "medication"
    INSURANCE_ID = "insurance_id"
    
    # Digital
    IP_ADDRESS = "ip_address"
    MAC_ADDRESS = "mac_address"
    DEVICE_ID = "device_id"
    
    # Biometric
    BIOMETRIC = "biometric"
    
    # Location
    GPS_COORDINATES = "gps_coordinates"
    
    # Custom
    CUSTOM = "custom"


@dataclass
class PIIMatch:
    """Represents a detected PII match.
    
    Security: Contains location and type for masking.
    """
    pii_type: PIIType
    value: str
    start: int
    end: int
    confidence: float
    
    def get_masked_value(self, strategy: str = "redact") -> str:
        """Get masked version of the PII value.
        
        Security: Multiple masking strategies for different compliance needs.
        """
        if strategy == "redact":
            return f"[{self.pii_type.value.upper()}]"
        elif strategy == "hash":
            return hashlib.sha256(self.value.encode()).hexdigest()[:16]
        elif strategy == "partial":
            if len(self.value) > 4:
                return self.value[:2] + "*" * (len(self.value) - 4) + self.value[-2:]
            return "*" * len(self.value)
        else:
            return "[REDACTED]"


class PIIDetector:
    """Detects PII in text using regex patterns and heuristics.
    
    Security: Uses compiled regex for performance and accuracy.
    Compliance: Patterns designed to meet regulatory requirements.
    """
    
    # Regex patterns for PII detection
    # Security: Patterns are comprehensive but may need tuning per use case
    PATTERNS: Dict[PIIType, List[re.Pattern]] = {
        PIIType.CREDIT_CARD: [
            re.compile(r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b'),
            re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        ],
        PIIType.SSN: [
            re.compile(r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'),
        ],
        PIIType.EMAIL: [
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        ],
        PIIType.PHONE: [
            re.compile(r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'),
            re.compile(r'\b\+?[0-9]{1,4}[-.\s]?[0-9]{2,4}[-.\s]?[0-9]{2,4}[-.\s]?[0-9]{2,4}\b'),
        ],
        PIIType.IP_ADDRESS: [
            re.compile(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'),
            re.compile(r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'),  # IPv6
        ],
        PIIType.BANK_ACCOUNT: [
            re.compile(r'\b[0-9]{8,17}\b'),  # Generic account number
        ],
        PIIType.ROUTING_NUMBER: [
            re.compile(r'\b[0-9]{9}\b'),  # US routing number
        ],
        PIIType.DOB: [
            re.compile(r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b'),
            re.compile(r'\b(?:19|20)\d{2}[/-](?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])\b'),
        ],
        PIIType.GPS_COORDINATES: [
            re.compile(r'\b[-+]?(?:[1-8]?\d(?:\.\d+)?|90(?:\.0+)?),\s*[-+]?(?:180(?:\.0+)?|(?:(?:1[0-7]\d)|(?:[1-9]?\d))(?:\.\d+)?)\b'),
        ],
        PIIType.CVV: [
            re.compile(r'\bCVV:?\s*\d{3,4}\b', re.IGNORECASE),
        ],
        PIIType.PASSPORT: [
            re.compile(r'\b[A-Z]{1,2}[0-9]{6,9}\b'),
        ],
    }
    
    # Keywords that indicate PII context
    CONTEXT_KEYWORDS: Dict[PIIType, List[str]] = {
        PIIType.CREDIT_CARD: ["card", "visa", "mastercard", "amex", "credit", "debit"],
        PIIType.SSN: ["ssn", "social security", "social-security"],
        PIIType.BANK_ACCOUNT: ["account", "checking", "savings", "iban", "swift"],
        PIIType.MEDICAL_RECORD: ["patient", "diagnosis", "treatment", "prescription", "medical"],
        PIIType.INSURANCE_ID: ["insurance", "policy", "member id", "group number"],
    }
    
    def __init__(self, pii_types: Optional[Set[PIIType]] = None):
        """Initialize the PII detector.
        
        Args:
            pii_types: Set of PII types to detect. If None, detects all types.
            
        Security: Allows limiting detection to specific PII types.
        """
        self.pii_types = pii_types or set(PIIType)
    
    def detect(self, text: str) -> List[PIIMatch]:
        """Detect all PII in the given text.
        
        Args:
            text: The text to scan for PII.
            
        Returns:
            List of PIIMatch objects for each detected PII.
            
        Security: Scans entire text for all configured PII types.
        """
        matches = []
        
        for pii_type in self.pii_types:
            if pii_type in self.PATTERNS:
                for pattern in self.PATTERNS[pii_type]:
                    for match in pattern.finditer(text):
                        confidence = self._calculate_confidence(pii_type, match.group(), text, match.start())
                        if confidence > 0.5:
                            matches.append(PIIMatch(
                                pii_type=pii_type,
                                value=match.group(),
                                start=match.start(),
                                end=match.end(),
                                confidence=confidence
                            ))
        
        # Remove overlapping matches, keeping highest confidence
        matches = self._remove_overlaps(matches)
        
        return matches
    
    def _calculate_confidence(self, pii_type: PIIType, value: str, text: str, position: int) -> float:
        """Calculate confidence score for a PII match.
        
        Security: Higher confidence = more likely to be actual PII.
        """
        confidence = 0.6  # Base confidence for pattern match
        
        # Check for context keywords
        context_start = max(0, position - 50)
        context_end = min(len(text), position + len(value) + 50)
        context = text[context_start:context_end].lower()
        
        if pii_type in self.CONTEXT_KEYWORDS:
            for keyword in self.CONTEXT_KEYWORDS[pii_type]:
                if keyword in context:
                    confidence += 0.2
                    break
        
        # Validate specific formats
        if pii_type == PIIType.CREDIT_CARD:
            if self._luhn_check(value.replace("-", "").replace(" ", "")):
                confidence += 0.3
        
        if pii_type == PIIType.EMAIL:
            if "@" in value and "." in value.split("@")[1]:
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    def _luhn_check(self, card_number: str) -> bool:
        """Validate credit card number using Luhn algorithm.
        
        Security: Reduces false positives for credit card detection.
        Compliance: Required for PCI-DSS.
        """
        try:
            digits = [int(d) for d in card_number if d.isdigit()]
            if len(digits) < 13 or len(digits) > 19:
                return False
            
            checksum = 0
            for i, digit in enumerate(reversed(digits)):
                if i % 2 == 1:
                    digit *= 2
                    if digit > 9:
                        digit -= 9
                checksum += digit
            
            return checksum % 10 == 0
        except (ValueError, IndexError):
            return False
    
    def _remove_overlaps(self, matches: List[PIIMatch]) -> List[PIIMatch]:
        """Remove overlapping matches, keeping highest confidence.
        
        Security: Prevents duplicate masking of same text.
        """
        if not matches:
            return matches
        
        # Sort by start position, then by confidence (descending)
        matches.sort(key=lambda m: (m.start, -m.confidence))
        
        result = []
        last_end = -1
        
        for match in matches:
            if match.start >= last_end:
                result.append(match)
                last_end = match.end
        
        return result
    
    def contains_pii(self, text: str) -> bool:
        """Quick check if text contains any PII.
        
        Security: Fast check for validation before processing.
        """
        return len(self.detect(text)) > 0
    
    def get_pii_summary(self, text: str) -> Dict[str, int]:
        """Get summary of PII types found in text.
        
        Security: For audit logging without exposing actual values.
        """
        matches = self.detect(text)
        summary = {}
        for match in matches:
            key = match.pii_type.value
            summary[key] = summary.get(key, 0) + 1
        return summary
