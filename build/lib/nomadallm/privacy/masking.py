"""
NomadaLLM Data Masking

Masks PII in text using various strategies to protect sensitive data
while maintaining text usability for LLM processing.

Security: Core component for data protection.
Compliance: Required for GDPR, HIPAA, PCI-DSS.
"""

import hashlib
import secrets
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from nomadallm.privacy.pii_detector import PIIDetector, PIIMatch, PIIType


@dataclass
class MaskingResult:
    """Result of masking operation.
    
    Security: Contains both masked text and mapping for restoration.
    """
    original_text: str
    masked_text: str
    mask_map: Dict[str, str]  # token -> original value
    pii_found: List[PIIMatch]
    
    def restore(self) -> str:
        """Restore original text from masked text.
        
        Security: Only call this when absolutely necessary.
        Compliance: Restoration should be audited.
        """
        result = self.masked_text
        for token, original in self.mask_map.items():
            result = result.replace(token, original)
        return result


class DataMasker:
    """Masks PII in text using configurable strategies.
    
    Security: Supports multiple masking strategies for different compliance needs.
    Compliance: Strategies designed for GDPR, HIPAA, PCI-DSS.
    """
    
    def __init__(
        self,
        strategy: str = "redact",
        pii_types: Optional[set] = None,
        reversible: bool = False
    ):
        """Initialize the data masker.
        
        Args:
            strategy: Masking strategy - redact, hash, tokenize, partial, encrypt
            pii_types: Set of PII types to mask. If None, masks all detected PII.
            reversible: If True, maintains mapping for restoration.
            
        Security: Reversible mode should only be used when restoration is required.
        """
        self.strategy = strategy
        self.reversible = reversible
        self.detector = PIIDetector(pii_types)
        self._token_counter = 0
    
    def mask(self, text: str) -> MaskingResult:
        """Mask all PII in the given text.
        
        Args:
            text: The text to mask.
            
        Returns:
            MaskingResult with masked text and metadata.
            
        Security: Detects and masks all configured PII types.
        """
        matches = self.detector.detect(text)
        
        if not matches:
            return MaskingResult(
                original_text=text,
                masked_text=text,
                mask_map={},
                pii_found=[]
            )
        
        # Sort matches by position (reverse order for replacement)
        matches.sort(key=lambda m: m.start, reverse=True)
        
        masked_text = text
        mask_map = {}
        
        for match in matches:
            token = self._generate_token(match)
            
            if self.reversible:
                mask_map[token] = match.value
            
            masked_text = (
                masked_text[:match.start] +
                token +
                masked_text[match.end:]
            )
        
        return MaskingResult(
            original_text=text if self.reversible else "",
            masked_text=masked_text,
            mask_map=mask_map,
            pii_found=matches
        )
    
    def _generate_token(self, match: PIIMatch) -> str:
        """Generate a masking token based on strategy.
        
        Security: Token generation varies by strategy.
        """
        if self.strategy == "redact":
            return f"[{match.pii_type.value.upper()}]"
        
        elif self.strategy == "hash":
            hash_value = hashlib.sha256(match.value.encode()).hexdigest()[:12]
            return f"[HASH:{hash_value}]"
        
        elif self.strategy == "tokenize":
            self._token_counter += 1
            token_id = f"{match.pii_type.value}_{self._token_counter}"
            return f"[TOKEN:{token_id}]"
        
        elif self.strategy == "partial":
            return self._partial_mask(match.value, match.pii_type)
        
        elif self.strategy == "placeholder":
            return self._get_placeholder(match.pii_type)
        
        else:
            return "[REDACTED]"
    
    def _partial_mask(self, value: str, pii_type: PIIType) -> str:
        """Partially mask a value, showing some characters.
        
        Security: Shows minimal characters for usability.
        Compliance: May not be suitable for all compliance requirements.
        """
        if pii_type == PIIType.EMAIL:
            parts = value.split("@")
            if len(parts) == 2:
                local = parts[0]
                domain = parts[1]
                if len(local) > 2:
                    masked_local = local[0] + "*" * (len(local) - 2) + local[-1]
                else:
                    masked_local = "*" * len(local)
                return f"{masked_local}@{domain}"
        
        elif pii_type == PIIType.CREDIT_CARD:
            digits = "".join(c for c in value if c.isdigit())
            if len(digits) >= 4:
                return "*" * (len(digits) - 4) + digits[-4:]
        
        elif pii_type == PIIType.PHONE:
            digits = "".join(c for c in value if c.isdigit())
            if len(digits) >= 4:
                return "*" * (len(digits) - 4) + digits[-4:]
        
        elif pii_type == PIIType.SSN:
            digits = "".join(c for c in value if c.isdigit())
            if len(digits) == 9:
                return f"***-**-{digits[-4:]}"
        
        # Default partial masking
        if len(value) > 4:
            return value[0] + "*" * (len(value) - 2) + value[-1]
        return "*" * len(value)
    
    def _get_placeholder(self, pii_type: PIIType) -> str:
        """Get a realistic placeholder for a PII type.
        
        Security: Placeholders maintain text structure without real data.
        """
        placeholders = {
            PIIType.EMAIL: "user@example.com",
            PIIType.PHONE: "(555) 555-0100",
            PIIType.CREDIT_CARD: "4111-1111-1111-1111",
            PIIType.SSN: "000-00-0000",
            PIIType.NAME: "John Doe",
            PIIType.ADDRESS: "123 Main St, Anytown, ST 12345",
            PIIType.DOB: "01/01/1990",
            PIIType.IP_ADDRESS: "192.0.2.1",
        }
        return placeholders.get(pii_type, "[PLACEHOLDER]")
    
    def unmask(self, result: MaskingResult) -> str:
        """Restore original text from a masking result.
        
        Security: Only works if masking was done with reversible=True.
        Compliance: Unmasking should be audited.
        """
        if not result.mask_map:
            return result.masked_text
        
        return result.restore()


class BatchMasker:
    """Masks PII across multiple texts with consistent tokenization.
    
    Security: Ensures same PII gets same token across batch.
    """
    
    def __init__(self, strategy: str = "tokenize"):
        self.strategy = strategy
        self.global_token_map: Dict[str, str] = {}
        self._counter = 0
    
    def mask_batch(self, texts: List[str]) -> List[MaskingResult]:
        """Mask multiple texts with consistent tokens.
        
        Security: Same PII value gets same token across all texts.
        """
        results = []
        detector = PIIDetector()
        
        for text in texts:
            matches = detector.detect(text)
            masked_text = text
            mask_map = {}
            
            # Sort by position (reverse)
            matches.sort(key=lambda m: m.start, reverse=True)
            
            for match in matches:
                # Check if we've seen this value before
                if match.value in self.global_token_map:
                    token = self.global_token_map[match.value]
                else:
                    self._counter += 1
                    token = f"[{match.pii_type.value.upper()}_{self._counter}]"
                    self.global_token_map[match.value] = token
                
                mask_map[token] = match.value
                masked_text = (
                    masked_text[:match.start] +
                    token +
                    masked_text[match.end:]
                )
            
            results.append(MaskingResult(
                original_text=text,
                masked_text=masked_text,
                mask_map=mask_map,
                pii_found=matches
            ))
        
        return results
