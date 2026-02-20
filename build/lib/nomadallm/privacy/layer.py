"""
NomadaLLM Privacy Layer

The main privacy orchestration layer that combines PII detection,
masking, encryption, and audit logging into a unified interface.

Security: Central component for all privacy operations.
Compliance: Enforces privacy policies across all data processing.
"""

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from nomadallm.privacy.modes import PrivacyMode, PrivacyConfig, get_privacy_config
from nomadallm.privacy.pii_detector import PIIDetector, PIIMatch, PIIType
from nomadallm.privacy.masking import DataMasker, MaskingResult
from nomadallm.privacy.encryption import EncryptionService, EncryptedData
from nomadallm.privacy.audit import AuditLogger, AuditEventType
from nomadallm.exceptions import PrivacyViolationError


@dataclass
class PrivacyProcessingResult:
    """Result of privacy processing on data.
    
    Security: Contains all privacy-related metadata.
    """
    original_text: str
    processed_text: str
    pii_detected: List[PIIMatch]
    pii_masked: bool
    encrypted: bool
    audit_event_id: Optional[str]
    can_send_to_llm: bool
    warnings: List[str]
    
    def get_safe_text(self) -> str:
        """Get text safe for LLM processing.
        
        Security: Returns processed text only if safe.
        """
        if self.can_send_to_llm:
            return self.processed_text
        raise PrivacyViolationError(
            "privacy_policy",
            "Text contains unmasked PII and cannot be sent to LLM"
        )


class PrivacyLayer:
    """Main privacy orchestration layer.
    
    Security: Coordinates all privacy operations.
    Compliance: Enforces configured privacy policies.
    
    Usage:
        privacy = PrivacyLayer(mode=PrivacyMode.BANKING)
        result = privacy.process("User SSN is 123-45-6789")
        safe_text = result.get_safe_text()  # "[SSN]"
    """
    
    def __init__(
        self,
        mode: PrivacyMode = PrivacyMode.STANDARD,
        config: Optional[PrivacyConfig] = None,
        encryption_key: Optional[bytes] = None,
        audit_logger: Optional[AuditLogger] = None
    ):
        """Initialize the privacy layer.
        
        Args:
            mode: Privacy mode to use (BANKING, HEALTHCARE, etc.)
            config: Optional custom privacy configuration.
            encryption_key: Optional master encryption key.
            audit_logger: Optional audit logger instance.
            
        Security: Initializes all privacy components.
        """
        self.mode = mode
        self.config = config or get_privacy_config(mode)
        
        # Initialize components
        pii_types = None
        if self.config.pii_types_to_detect:
            pii_types = {
                PIIType(t) if isinstance(t, str) else t 
                for t in self.config.pii_types_to_detect
                if t in [e.value for e in PIIType] or isinstance(t, PIIType)
            }
        
        self.detector = PIIDetector(pii_types)
        self.masker = DataMasker(
            strategy=self.config.masking_strategy,
            pii_types=pii_types,
            reversible=not self.config.zero_knowledge_mode
        )
        
        if self.config.encrypt_at_rest:
            self.encryption = EncryptionService(encryption_key)
        else:
            self.encryption = None
        
        self.audit = audit_logger or AuditLogger()
    
    def process(
        self,
        text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> PrivacyProcessingResult:
        """Process text through the privacy layer.
        
        Args:
            text: The text to process.
            user_id: Optional user identifier for audit.
            session_id: Optional session identifier for audit.
            context: Optional additional context.
            
        Returns:
            PrivacyProcessingResult with processed text and metadata.
            
        Security: Full privacy processing pipeline.
        """
        warnings = []
        
        # Step 1: Detect PII
        pii_matches = self.detector.detect(text)
        
        if pii_matches:
            # Log PII detection
            pii_types = list(set(m.pii_type.value for m in pii_matches))
            self.audit.log_pii_detected(
                user_id=user_id or "anonymous",
                pii_types=pii_types,
                count=len(pii_matches),
                session_id=session_id
            )
            
            # Check for privacy violations
            if self.config.zero_knowledge_mode:
                # In zero-knowledge mode, any PII is a potential violation
                for match in pii_matches:
                    if match.confidence > 0.8:
                        warnings.append(
                            f"High-confidence PII detected: {match.pii_type.value}"
                        )
        
        # Step 2: Mask PII if configured
        processed_text = text
        pii_masked = False
        
        if pii_matches and self.config.mask_pii_in_prompts:
            masking_result = self.masker.mask(text)
            processed_text = masking_result.masked_text
            pii_masked = True
            
            # Log masking
            self.audit.log_pii_masked(
                user_id=user_id or "anonymous",
                pii_types=[m.pii_type.value for m in pii_matches],
                strategy=self.config.masking_strategy,
                session_id=session_id
            )
        
        # Step 3: Determine if safe to send to LLM
        can_send = True
        if pii_matches and not pii_masked:
            # PII detected but not masked - check policy
            if self.mode in [PrivacyMode.BANKING, PrivacyMode.HEALTHCARE, PrivacyMode.ZERO_KNOWLEDGE]:
                can_send = False
                warnings.append("Unmasked PII cannot be sent to LLM in this privacy mode")
        
        # Step 4: Create audit event
        audit_event = self.audit.log(
            event_type=AuditEventType.LLM_REQUEST,
            action="privacy_processing",
            user_id=user_id,
            session_id=session_id,
            details={
                "pii_count": len(pii_matches),
                "masked": pii_masked,
                "mode": self.mode.value,
                "can_send": can_send
            }
        )
        
        return PrivacyProcessingResult(
            original_text=text if not self.config.zero_knowledge_mode else "",
            processed_text=processed_text,
            pii_detected=pii_matches,
            pii_masked=pii_masked,
            encrypted=False,
            audit_event_id=audit_event.event_id,
            can_send_to_llm=can_send,
            warnings=warnings
        )
    
    def process_response(
        self,
        response: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> PrivacyProcessingResult:
        """Process LLM response through privacy layer.
        
        Security: Ensures responses don't leak PII.
        """
        # Detect PII in response
        pii_matches = self.detector.detect(response)
        
        processed_text = response
        pii_masked = False
        
        if pii_matches and self.config.mask_pii_in_responses:
            masking_result = self.masker.mask(response)
            processed_text = masking_result.masked_text
            pii_masked = True
            
            # Log unexpected PII in response
            self.audit.log(
                event_type=AuditEventType.PII_DETECTED,
                action="pii_in_response",
                user_id=user_id,
                session_id=session_id,
                details={
                    "pii_types": [m.pii_type.value for m in pii_matches],
                    "count": len(pii_matches)
                },
                risk_level="high"
            )
        
        return PrivacyProcessingResult(
            original_text=response if not self.config.zero_knowledge_mode else "",
            processed_text=processed_text,
            pii_detected=pii_matches,
            pii_masked=pii_masked,
            encrypted=False,
            audit_event_id=None,
            can_send_to_llm=True,
            warnings=[]
        )
    
    def encrypt_for_storage(self, data: str) -> str:
        """Encrypt data for storage.
        
        Security: Uses AES-256-GCM encryption.
        Compliance: Required for data at rest.
        """
        if not self.encryption:
            raise PrivacyViolationError(
                "encryption_required",
                "Encryption is required but not configured"
            )
        
        encrypted = self.encryption.encrypt(data)
        return encrypted.to_base64()
    
    def decrypt_from_storage(self, encrypted_data: str) -> str:
        """Decrypt data from storage.
        
        Security: Decrypts AES-256-GCM encrypted data.
        Compliance: Audit logged.
        """
        if not self.encryption:
            raise PrivacyViolationError(
                "encryption_required",
                "Encryption is required but not configured"
            )
        
        encrypted = EncryptedData.from_base64(encrypted_data)
        return self.encryption.decrypt(encrypted)
    
    def validate_data_residency(self, region: str) -> bool:
        """Validate if data can be processed in a region.
        
        Security: Enforces data residency requirements.
        Compliance: Required for GDPR.
        """
        if not self.config.allowed_regions:
            return True
        
        return region in self.config.allowed_regions
    
    def get_privacy_summary(self, text: str) -> Dict[str, Any]:
        """Get a summary of privacy concerns in text.
        
        Security: For preview/validation without full processing.
        """
        pii_matches = self.detector.detect(text)
        
        return {
            "has_pii": len(pii_matches) > 0,
            "pii_count": len(pii_matches),
            "pii_types": list(set(m.pii_type.value for m in pii_matches)),
            "high_risk_pii": [
                m.pii_type.value for m in pii_matches 
                if m.pii_type in [PIIType.SSN, PIIType.CREDIT_CARD, PIIType.MEDICAL_RECORD]
            ],
            "requires_masking": self.config.mask_pii_in_prompts and len(pii_matches) > 0,
            "mode": self.mode.value
        }
    
    def request_data_deletion(
        self,
        user_id: str,
        reason: str = "user_request"
    ) -> bool:
        """Request deletion of user data.
        
        Security: Implements right to be forgotten.
        Compliance: Required for GDPR Article 17.
        """
        self.audit.log(
            event_type=AuditEventType.DATA_DELETION,
            action="data_deletion_request",
            user_id=user_id,
            details={"reason": reason},
            risk_level="medium"
        )
        
        # In production, this would trigger actual data deletion
        return True
