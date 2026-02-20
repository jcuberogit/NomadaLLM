"""
NomadaLLM OpenAI Provider

OpenAI GPT models integration.

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


class OpenAIProvider(Provider):
    """OpenAI GPT provider.
    
    Security: Uses official OpenAI SDK with secure defaults.
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.OPENAI,
                api_key=os.getenv("OPENAI_API_KEY"),
                model="gpt-4o"
            )
        super().__init__(config)
        self._client = None
    
    def _validate_config(self) -> None:
        """Validate OpenAI configuration."""
        if not self.config.api_key:
            raise AuthenticationError("OPENAI_API_KEY not configured")
    
    def _get_client(self):
        """Get or create OpenAI client.
        
        Security: Lazy initialization to avoid import errors.
        """
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ProviderError(
                    "openai",
                    Exception("openai package not installed. Run: pip install openai")
                )
        return self._client
    
    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using OpenAI.
        
        Security: Does not log message content.
        """
        client = self._get_client()
        start_time = time.time()
        
        try:
            response = await client.chat.completions.create(
                model=kwargs.get("model", self.config.model),
                messages=[m.to_dict() for m in messages],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "temperature"]}
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return CompletionResponse(
                content=response.choices[0].message.content or "",
                model=response.model,
                provider="openai",
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
                total_tokens=response.usage.total_tokens if response.usage else 0,
                finish_reason=response.choices[0].finish_reason or "stop",
                latency_ms=latency_ms
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str:
                raise RateLimitError()
            elif "authentication" in error_str or "api key" in error_str:
                raise AuthenticationError("Invalid API key")
            else:
                raise ProviderError("openai", e)
    
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[CompletionResponse]:
        """Stream completion using OpenAI.
        
        Security: Does not log streamed content.
        """
        client = self._get_client()
        start_time = time.time()
        
        try:
            stream = await client.chat.completions.create(
                model=kwargs.get("model", self.config.model),
                messages=[m.to_dict() for m in messages],
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                temperature=kwargs.get("temperature", self.config.temperature),
                stream=True,
                **{k: v for k, v in kwargs.items() if k not in ["model", "max_tokens", "temperature"]}
            )
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    latency_ms = (time.time() - start_time) * 1000
                    yield CompletionResponse(
                        content=chunk.choices[0].delta.content,
                        model=chunk.model,
                        provider="openai",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        finish_reason=chunk.choices[0].finish_reason or "",
                        latency_ms=latency_ms,
                        is_complete=chunk.choices[0].finish_reason is not None
                    )
                    
        except Exception as e:
            raise ProviderError("openai", e)
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken.
        
        Security: Local token counting, no API call.
        """
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model(self.config.model)
            return len(encoding.encode(text))
        except ImportError:
            # Fallback: rough estimate
            return len(text) // 4
    
    async def health_check(self) -> bool:
        """Check OpenAI API availability.
        
        Security: Minimal API call to verify connectivity.
        """
        try:
            client = self._get_client()
            await client.models.list()
            return True
        except Exception:
            return False
