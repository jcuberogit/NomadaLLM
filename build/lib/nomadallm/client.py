"""
NomadaLLM Client

The main client class that provides a unified interface for LLM interactions
with built-in privacy controls.

Security: All requests pass through the privacy layer.
Compliance: Automatic PII detection, masking, and audit logging.
"""

import os
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from nomadallm.privacy import PrivacyLayer, PrivacyMode
from nomadallm.providers.base import Provider, ProviderConfig, ProviderType, Message, CompletionResponse
from nomadallm.providers.openai import OpenAIProvider
from nomadallm.providers.anthropic import AnthropicProvider
from nomadallm.providers.gemini import GeminiProvider
from nomadallm.providers.local import LocalProvider
from nomadallm.providers.embedded import EmbeddedProvider
from nomadallm.licensing import LicenseValidator, LicenseInfo, UsageTracker
from nomadallm.exceptions import NomadaLLMError, ProviderError, UsageLimitError


class NomadaLLM:
    """Universal LLM client with privacy-first architecture.
    
    Security: All data passes through privacy layer before reaching LLM.
    Compliance: Automatic PII handling for GDPR, HIPAA, PCI-DSS, SOC2.
    
    Usage:
        # Simple usage
        llm = NomadaLLM(api_key="xxx", privacy_mode="banking")
        response = await llm.chat("Analyze this account")
        
        # With explicit provider
        llm = NomadaLLM(provider="anthropic", privacy_mode="healthcare")
        response = await llm.chat("Patient symptoms include...")
        
        # Local/private mode
        llm = NomadaLLM(provider="local", privacy_mode="zero_knowledge")
        response = await llm.chat("Sensitive data here")
    """
    
    PROVIDER_MAP = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "gemini": GeminiProvider,
        "google": GeminiProvider,
        "local": LocalProvider,
        "ollama": LocalProvider,
        "embedded": EmbeddedProvider,
        "offline": EmbeddedProvider,
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "gemini",
        model: Optional[str] = None,
        privacy_mode: Union[str, PrivacyMode] = PrivacyMode.STANDARD,
        license_key: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize NomadaLLM client.
        
        Args:
            api_key: API key for the provider. If None, reads from environment.
            provider: LLM provider - openai, anthropic, gemini, local.
            model: Model name. If None, uses provider default.
            privacy_mode: Privacy mode - standard, banking, healthcare, enterprise, zero_knowledge.
            license_key: NomadaLLM license key. If None, uses free tier (100 calls/day).
            user_id: Optional user identifier for audit logging.
            session_id: Optional session identifier for audit logging.
            **kwargs: Additional provider-specific configuration.
            
        Security: API keys should be passed via environment variables in production.
        
        Pricing Tiers (all features included in every tier):
            - Free: 100 calls/day
            - Indie ($9/mo): 10,000 calls/day
            - Pro ($29/mo): 100,000 calls/day
            - Enterprise ($99/mo): Unlimited
        """
        self.user_id = user_id
        self.session_id = session_id
        
        # Initialize licensing
        license_key = license_key or os.environ.get("NOMADALLM_LICENSE_KEY")
        self._license_validator = LicenseValidator()
        self._license_info = self._license_validator.validate(license_key)
        self._usage_tracker = UsageTracker(tier=self._license_info.tier)
        
        # Initialize privacy layer
        if isinstance(privacy_mode, str):
            privacy_mode = PrivacyMode(privacy_mode)
        self.privacy = PrivacyLayer(mode=privacy_mode)
        
        # Initialize provider
        self._init_provider(provider, api_key, model, **kwargs)
        
        # Conversation history
        self._messages: List[Message] = []
        self._system_message: Optional[str] = None
    
    def _init_provider(
        self,
        provider: str,
        api_key: Optional[str],
        model: Optional[str],
        **kwargs
    ) -> None:
        """Initialize the LLM provider.
        
        Security: Validates provider configuration.
        """
        provider_lower = provider.lower()
        
        if provider_lower not in self.PROVIDER_MAP:
            raise NomadaLLMError(
                f"Unknown provider: {provider}",
                f"Supported providers: {list(self.PROVIDER_MAP.keys())}"
            )
        
        # Build provider config
        provider_type = {
            "openai": ProviderType.OPENAI,
            "anthropic": ProviderType.ANTHROPIC,
            "gemini": ProviderType.GEMINI,
            "google": ProviderType.GEMINI,
            "local": ProviderType.LOCAL,
            "ollama": ProviderType.LOCAL,
            "embedded": ProviderType.LOCAL,
            "offline": ProviderType.LOCAL,
        }[provider_lower]
        
        config = ProviderConfig(
            provider_type=provider_type,
            api_key=api_key,
            model=model or "",
            **{k: v for k, v in kwargs.items() if hasattr(ProviderConfig, k)}
        )
        
        # Create provider instance
        provider_class = self.PROVIDER_MAP[provider_lower]
        self.provider: Provider = provider_class(config)
    
    def set_system_message(self, message: str) -> 'NomadaLLM':
        """Set the system message for conversations.
        
        Args:
            message: System message/instructions for the LLM.
            
        Returns:
            Self for method chaining.
            
        Security: System message is also processed through privacy layer.
        """
        # Process system message through privacy layer
        result = self.privacy.process(
            message,
            user_id=self.user_id,
            session_id=self.session_id
        )
        self._system_message = result.processed_text
        return self
    
    async def prompt(
        self,
        message: str,
        user_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Send a prompt and get a response.
        
        Args:
            message: Prompt to send for reasoning/decision making.
            user_data: Optional additional user data to include.
            **kwargs: Additional provider-specific options.
            
        Returns:
            LLM response text.
            
        Security: Message is processed through privacy layer.
        
        Raises:
            UsageLimitError: If daily call limit has been reached.
        """
        # Check usage limits
        if not self._usage_tracker.can_call():
            usage = self._usage_tracker.get_usage()
            raise UsageLimitError(
                calls_today=usage.calls_today,
                daily_limit=usage.daily_limit,
                tier=usage.tier.value
            )
        
        # Record the call
        self._usage_tracker.record_call()
        
        # Process message through privacy layer
        privacy_result = self.privacy.process(
            message,
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        # Get safe text for LLM
        safe_message = privacy_result.get_safe_text()
        
        # Build messages list
        messages = []
        
        if self._system_message:
            messages.append(Message(role="system", content=self._system_message))
        
        # Add conversation history
        messages.extend(self._messages)
        
        # Add current message
        messages.append(Message(role="user", content=safe_message))
        
        # Get completion from provider
        response = await self.provider.complete(messages, **kwargs)
        
        # Process response through privacy layer
        response_result = self.privacy.process_response(
            response.content,
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        # Update conversation history
        self._messages.append(Message(role="user", content=safe_message))
        self._messages.append(Message(role="assistant", content=response_result.processed_text))
        
        return response_result.processed_text
    
    async def chat(
        self,
        message: str,
        user_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Alias for prompt() - for conversational use cases.
        
        Deprecated: Use prompt() for embedded intelligence.
        """
        return await self.prompt(message, user_data, **kwargs)
    
    async def chat_stream(
        self,
        message: str,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream a chat response.
        
        Args:
            message: User message to send.
            **kwargs: Additional provider-specific options.
            
        Yields:
            Response text chunks.
            
        Security: Each chunk is processed through privacy layer.
        """
        # Process message through privacy layer
        privacy_result = self.privacy.process(
            message,
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        safe_message = privacy_result.get_safe_text()
        
        # Build messages list
        messages = []
        
        if self._system_message:
            messages.append(Message(role="system", content=self._system_message))
        
        messages.extend(self._messages)
        messages.append(Message(role="user", content=safe_message))
        
        # Stream from provider
        full_response = ""
        async for chunk in self.provider.stream(messages, **kwargs):
            # Process each chunk through privacy layer
            chunk_result = self.privacy.process_response(
                chunk.content,
                user_id=self.user_id,
                session_id=self.session_id
            )
            full_response += chunk_result.processed_text
            yield chunk_result.processed_text
        
        # Update conversation history
        self._messages.append(Message(role="user", content=safe_message))
        self._messages.append(Message(role="assistant", content=full_response))
    
    async def complete(
        self,
        prompt: str,
        **kwargs
    ) -> CompletionResponse:
        """Get a raw completion (no conversation history).
        
        Args:
            prompt: The prompt to complete.
            **kwargs: Additional provider-specific options.
            
        Returns:
            Full CompletionResponse with metadata.
            
        Security: Prompt is processed through privacy layer.
        """
        # Process prompt through privacy layer
        privacy_result = self.privacy.process(
            prompt,
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        messages = [Message(role="user", content=privacy_result.get_safe_text())]
        
        if self._system_message:
            messages.insert(0, Message(role="system", content=self._system_message))
        
        response = await self.provider.complete(messages, **kwargs)
        
        # Process response
        response_result = self.privacy.process_response(
            response.content,
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        response.content = response_result.processed_text
        return response
    
    def clear_history(self) -> 'NomadaLLM':
        """Clear conversation history.
        
        Returns:
            Self for method chaining.
        """
        self._messages = []
        return self
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history.
        
        Returns:
            List of message dictionaries.
            
        Security: Returns processed (masked) messages only.
        """
        return [m.to_dict() for m in self._messages]
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text.
        
        Args:
            text: Text to count tokens for.
            
        Returns:
            Token count.
        """
        return await self.provider.count_tokens(text)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components.
        
        Returns:
            Health status dictionary.
        """
        provider_healthy = await self.provider.health_check()
        
        return {
            "healthy": provider_healthy,
            "provider": self.provider.get_provider_name(),
            "model": self.provider.get_model_name(),
            "privacy_mode": self.privacy.mode.value,
        }
    
    def get_privacy_summary(self, text: str) -> Dict[str, Any]:
        """Get privacy analysis of text without processing.
        
        Args:
            text: Text to analyze.
            
        Returns:
            Privacy summary dictionary.
        """
        return self.privacy.get_privacy_summary(text)
    
    def get_usage(self) -> Dict[str, Any]:
        """Get current usage statistics.
        
        Returns:
            Dictionary with usage info including calls_today, calls_remaining, daily_limit.
        """
        return self._usage_tracker.get_usage().to_dict()
    
    def get_license_info(self) -> Dict[str, Any]:
        """Get license information.
        
        Returns:
            Dictionary with license info including tier, expires_at, features.
        """
        return self._license_info.to_dict()
    
    def __repr__(self) -> str:
        return (
            f"NomadaLLM(provider={self.provider.get_provider_name()!r}, "
            f"model={self.provider.get_model_name()!r}, "
            f"privacy_mode={self.privacy.mode.value!r}, "
            f"tier={self._license_info.tier.value!r})"
        )
