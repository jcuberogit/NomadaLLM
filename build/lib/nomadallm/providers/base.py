"""
NomadaLLM Provider Base

Abstract base class for LLM providers with consistent interface.

Security: All providers must implement privacy-aware methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum


class ProviderType(Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class ProviderConfig:
    """Configuration for an LLM provider.
    
    Security: API keys should be loaded from environment variables.
    """
    provider_type: ProviderType
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    
    # Privacy settings
    log_requests: bool = False
    log_responses: bool = False
    
    # Custom headers
    headers: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.model:
            self.model = self._get_default_model()
    
    def _get_default_model(self) -> str:
        """Get default model for provider."""
        defaults = {
            ProviderType.OPENAI: "gpt-4o",
            ProviderType.ANTHROPIC: "claude-sonnet-4-20250514",
            ProviderType.GEMINI: "gemini-2.0-flash",
            ProviderType.LOCAL: "llama3",
        }
        return defaults.get(self.provider_type, "")


@dataclass
class Message:
    """A chat message."""
    role: str  # system, user, assistant
    content: str
    name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class CompletionResponse:
    """Response from LLM completion.
    
    Security: Contains usage metrics for billing and audit.
    """
    content: str
    model: str
    provider: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    finish_reason: str
    latency_ms: float
    
    # For streaming
    is_complete: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "provider": self.provider,
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
        }


class Provider(ABC):
    """Abstract base class for LLM providers.
    
    Security: Defines consistent interface for all providers.
    """
    
    def __init__(self, config: ProviderConfig):
        """Initialize provider with configuration.
        
        Security: Validates configuration before use.
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate provider configuration.
        
        Security: Ensures required settings are present.
        """
        pass
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> CompletionResponse:
        """Generate a completion from messages.
        
        Args:
            messages: List of chat messages.
            **kwargs: Additional provider-specific options.
            
        Returns:
            CompletionResponse with generated text.
            
        Security: Must not log sensitive content.
        """
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[CompletionResponse]:
        """Stream a completion from messages.
        
        Args:
            messages: List of chat messages.
            **kwargs: Additional provider-specific options.
            
        Yields:
            CompletionResponse chunks.
            
        Security: Must not log sensitive content.
        """
        pass
    
    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Security: For cost estimation and limit enforcement.
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is available.
        
        Security: Does not expose internal errors.
        """
        pass
    
    def get_provider_name(self) -> str:
        """Get the provider name."""
        return self.config.provider_type.value
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.config.model
