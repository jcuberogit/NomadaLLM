"""
NomadaLLM Anthropic Provider

Anthropic Claude models integration.

Security: API key loaded from environment, never logged.
"""

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from nomadallm.providers.base import (
    Provider, ProviderConfig, ProviderType,
    Message, CompletionResponse
)
from nomadallm.exceptions import ProviderError, AuthenticationError, RateLimitError


class AnthropicProvider(Provider):
    """Anthropic Claude provider.
    
    Security: Uses official Anthropic SDK with secure defaults.
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.ANTHROPIC,
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                model="claude-sonnet-4-20250514"
            )
        super().__init__(config)
        self._client = None
    
    def _validate_config(self) -> None:
        """Validate Anthropic configuration."""
        if not self.config.api_key:
            raise AuthenticationError("ANTHROPIC_API_KEY not configured")
    
    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ProviderError(
                    "anthropic",
                    Exception("anthropic package not installed. Run: pip install anthropic")
                )
        return self._client
    
    def _convert_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Anthropic format.
        
        Security: Separates system message from conversation.
        """
        system_message = ""
        conversation = []
        
        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                conversation.append({
                    "role": msg.role,
                    "content": msg.content
                })
        
        return system_message, conversation
    
    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using Anthropic."""
        client = self._get_client()
        start_time = time.time()
        
        system_message, conversation = self._convert_messages(messages)
        
        try:
            response = await client.messages.create(
                model=kwargs.get("model", self.config.model),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                system=system_message if system_message else None,
                messages=conversation,
                temperature=kwargs.get("temperature", self.config.temperature),
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = ""
            if response.content:
                content = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
            
            return CompletionResponse(
                content=content,
                model=response.model,
                provider="anthropic",
                prompt_tokens=response.usage.input_tokens if response.usage else 0,
                completion_tokens=response.usage.output_tokens if response.usage else 0,
                total_tokens=(response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0,
                finish_reason=response.stop_reason or "stop",
                latency_ms=latency_ms
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str:
                raise RateLimitError()
            elif "authentication" in error_str or "api key" in error_str:
                raise AuthenticationError("Invalid API key")
            else:
                raise ProviderError("anthropic", e)
    
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[CompletionResponse]:
        """Stream completion using Anthropic."""
        client = self._get_client()
        start_time = time.time()
        
        system_message, conversation = self._convert_messages(messages)
        
        try:
            async with client.messages.stream(
                model=kwargs.get("model", self.config.model),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                system=system_message if system_message else None,
                messages=conversation,
                temperature=kwargs.get("temperature", self.config.temperature),
            ) as stream:
                async for text in stream.text_stream:
                    latency_ms = (time.time() - start_time) * 1000
                    yield CompletionResponse(
                        content=text,
                        model=self.config.model,
                        provider="anthropic",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        finish_reason="",
                        latency_ms=latency_ms,
                        is_complete=False
                    )
                    
        except Exception as e:
            raise ProviderError("anthropic", e)
    
    async def count_tokens(self, text: str) -> int:
        """Estimate token count for Anthropic."""
        # Anthropic uses similar tokenization to GPT
        return len(text) // 4
    
    async def health_check(self) -> bool:
        """Check Anthropic API availability."""
        try:
            client = self._get_client()
            # Simple test message
            await client.messages.create(
                model=self.config.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}]
            )
            return True
        except Exception:
            return False
