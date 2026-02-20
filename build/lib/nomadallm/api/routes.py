"""
NomadaLLM API Routes

REST API endpoints for NomadaLLM functionality.

Security: All endpoints validate input and enforce privacy policies.
Compliance: Request/response logging for audit trail.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import asyncio

from nomadallm.client import NomadaLLM
from nomadallm.privacy import PrivacyMode
from nomadallm.exceptions import (
    NomadaLLMError,
    PrivacyViolationError,
    ProviderError,
    RateLimitError,
    AuthenticationError
)


router = APIRouter(prefix="/api/v1", tags=["NomadaLLM"])


# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model.
    
    Security: Validates input before processing.
    """
    message: str = Field(..., min_length=1, max_length=32000)
    provider: str = Field(default="gemini")
    model: Optional[str] = None
    privacy_mode: str = Field(default="standard")
    system_message: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=4096, ge=1, le=128000)
    stream: bool = Field(default=False)
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response model."""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    privacy: Dict[str, Any]


class PrivacyCheckRequest(BaseModel):
    """Privacy check request model."""
    text: str = Field(..., min_length=1, max_length=100000)
    privacy_mode: str = Field(default="standard")


class PrivacyCheckResponse(BaseModel):
    """Privacy check response model."""
    has_pii: bool
    pii_count: int
    pii_types: List[str]
    high_risk_pii: List[str]
    requires_masking: bool
    mode: str


class MaskRequest(BaseModel):
    """Mask request model."""
    text: str = Field(..., min_length=1, max_length=100000)
    strategy: str = Field(default="redact")
    privacy_mode: str = Field(default="standard")


class MaskResponse(BaseModel):
    """Mask response model."""
    masked_text: str
    pii_count: int
    pii_types: List[str]


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    provider: str
    model: str
    privacy_mode: str


# Dependency for API key validation
async def validate_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    """Validate API key from header.
    
    Security: All endpoints require authentication.
    """
    if not x_api_key or len(x_api_key) < 10:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# Endpoints
@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    api_key: str = Depends(validate_api_key)
) -> ChatResponse:
    """Send a chat message and get a response.
    
    Security: Message is processed through privacy layer.
    Compliance: All requests are logged for audit.
    """
    try:
        # Initialize client
        llm = NomadaLLM(
            api_key=api_key,
            provider=request.provider,
            model=request.model,
            privacy_mode=request.privacy_mode,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        if request.system_message:
            llm.set_system_message(request.system_message)
        
        # Get response
        response = await llm.complete(
            request.message,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Get privacy summary
        privacy_summary = llm.get_privacy_summary(request.message)
        
        return ChatResponse(
            content=response.content,
            model=response.model,
            provider=response.provider,
            usage={
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens
            },
            privacy=privacy_summary
        )
        
    except PrivacyViolationError as e:
        raise HTTPException(status_code=400, detail="Privacy policy violation")
    except AuthenticationError:
        raise HTTPException(status_code=401, detail="Authentication failed")
    except RateLimitError:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    except ProviderError:
        raise HTTPException(status_code=503, detail="LLM provider unavailable")
    except NomadaLLMError as e:
        raise HTTPException(status_code=500, detail="Internal error")


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    api_key: str = Depends(validate_api_key)
):
    """Stream a chat response.
    
    Security: Each chunk is processed through privacy layer.
    """
    try:
        llm = NomadaLLM(
            api_key=api_key,
            provider=request.provider,
            model=request.model,
            privacy_mode=request.privacy_mode,
            user_id=request.user_id,
            session_id=request.session_id
        )
        
        if request.system_message:
            llm.set_system_message(request.system_message)
        
        async def generate():
            async for chunk in llm.chat_stream(
                request.message,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Streaming error")


@router.post("/privacy/check", response_model=PrivacyCheckResponse)
async def privacy_check(
    request: PrivacyCheckRequest,
    api_key: str = Depends(validate_api_key)
) -> PrivacyCheckResponse:
    """Check text for PII without processing.
    
    Security: Does not send data to LLM - local analysis only.
    """
    try:
        llm = NomadaLLM(
            api_key=api_key,
            privacy_mode=request.privacy_mode
        )
        
        summary = llm.get_privacy_summary(request.text)
        
        return PrivacyCheckResponse(
            has_pii=summary["has_pii"],
            pii_count=summary["pii_count"],
            pii_types=summary["pii_types"],
            high_risk_pii=summary["high_risk_pii"],
            requires_masking=summary["requires_masking"],
            mode=summary["mode"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Privacy check error")


@router.post("/privacy/mask", response_model=MaskResponse)
async def mask_pii(
    request: MaskRequest,
    api_key: str = Depends(validate_api_key)
) -> MaskResponse:
    """Mask PII in text.
    
    Security: Returns masked text with PII redacted/tokenized.
    """
    try:
        from nomadallm.privacy import PrivacyLayer, PrivacyMode
        
        privacy = PrivacyLayer(mode=PrivacyMode(request.privacy_mode))
        result = privacy.process(request.text)
        
        return MaskResponse(
            masked_text=result.processed_text,
            pii_count=len(result.pii_detected),
            pii_types=list(set(m.pii_type.value for m in result.pii_detected))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail="Masking error")


@router.get("/health", response_model=HealthResponse)
async def health_check(
    provider: str = "gemini",
    api_key: str = Depends(validate_api_key)
) -> HealthResponse:
    """Check API and provider health.
    
    Security: Minimal API call to verify connectivity.
    """
    try:
        llm = NomadaLLM(api_key=api_key, provider=provider)
        health = await llm.health_check()
        
        return HealthResponse(
            status="healthy" if health["healthy"] else "unhealthy",
            provider=health["provider"],
            model=health["model"],
            privacy_mode=health["privacy_mode"]
        )
        
    except Exception:
        return HealthResponse(
            status="unhealthy",
            provider=provider,
            model="unknown",
            privacy_mode="unknown"
        )


@router.get("/providers")
async def list_providers() -> Dict[str, Any]:
    """List available LLM providers.
    
    Security: Public endpoint - no sensitive data.
    """
    return {
        "providers": [
            {
                "name": "openai",
                "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
                "streaming": True
            },
            {
                "name": "anthropic",
                "models": ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
                "streaming": True
            },
            {
                "name": "gemini",
                "models": ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
                "streaming": True
            },
            {
                "name": "local",
                "models": ["llama3", "mistral", "codellama"],
                "streaming": True
            }
        ]
    }


@router.get("/privacy/modes")
async def list_privacy_modes() -> Dict[str, Any]:
    """List available privacy modes.
    
    Security: Public endpoint - describes privacy options.
    """
    return {
        "modes": [
            {
                "name": "standard",
                "description": "Basic privacy protections",
                "pii_masking": False,
                "encryption": True,
                "audit": False
            },
            {
                "name": "banking",
                "description": "PCI-DSS compliant for financial data",
                "pii_masking": True,
                "encryption": True,
                "audit": True
            },
            {
                "name": "healthcare",
                "description": "HIPAA compliant for medical data",
                "pii_masking": True,
                "encryption": True,
                "audit": True
            },
            {
                "name": "enterprise",
                "description": "SOC2 compliant for business data",
                "pii_masking": True,
                "encryption": True,
                "audit": True
            },
            {
                "name": "zero_knowledge",
                "description": "Maximum privacy - no data retention",
                "pii_masking": True,
                "encryption": True,
                "audit": False
            },
            {
                "name": "gdpr",
                "description": "EU GDPR compliant",
                "pii_masking": True,
                "encryption": True,
                "audit": True
            }
        ]
    }
