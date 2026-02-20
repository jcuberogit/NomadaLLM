"""
NomadaLLM Gemini Provider

Google Gemini models integration.

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


class GeminiProvider(Provider):
    """Google Gemini provider.
    
    Security: Uses official Google GenAI SDK.
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.GEMINI,
                api_key=os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
                model="gemini-2.0-flash"
            )
        super().__init__(config)
        self._client = None
    
    def _validate_config(self) -> None:
        """Validate Gemini configuration."""
        if not self.config.api_key:
            raise AuthenticationError("GOOGLE_API_KEY or GEMINI_API_KEY not configured")
    
    def _get_client(self):
        """Get or create Gemini client."""
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.config.api_key)
            except ImportError:
                raise ProviderError(
                    "gemini",
                    Exception("google-genai package not installed. Run: pip install google-genai")
                )
        return self._client
    
    def _convert_messages(self, messages: List[Message]) -> tuple:
        """Convert messages to Gemini format."""
        system_instruction = None
        contents = []
        
        for msg in messages:
            if msg.role == "system":
                system_instruction = msg.content
            else:
                role = "user" if msg.role == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg.content}]
                })
        
        return system_instruction, contents
    
    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using Gemini."""
        client = self._get_client()
        start_time = time.time()
        
        system_instruction, contents = self._convert_messages(messages)
        
        try:
            from google.genai import types
            
            config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", self.config.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                system_instruction=system_instruction,
            )
            
            response = await client.aio.models.generate_content(
                model=kwargs.get("model", self.config.model),
                contents=contents,
                config=config
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            content = response.text if response.text else ""
            
            # Extract usage if available
            prompt_tokens = 0
            completion_tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
                completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
            
            return CompletionResponse(
                content=content,
                model=self.config.model,
                provider="gemini",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                finish_reason="stop",
                latency_ms=latency_ms
            )
            
        except Exception as e:
            error_str = str(e).lower()
            if "rate limit" in error_str or "quota" in error_str:
                raise RateLimitError()
            elif "api key" in error_str or "authentication" in error_str:
                raise AuthenticationError("Invalid API key")
            else:
                raise ProviderError("gemini", e)
    
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[CompletionResponse]:
        """Stream completion using Gemini."""
        client = self._get_client()
        start_time = time.time()
        
        system_instruction, contents = self._convert_messages(messages)
        
        try:
            from google.genai import types
            
            config = types.GenerateContentConfig(
                temperature=kwargs.get("temperature", self.config.temperature),
                max_output_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                system_instruction=system_instruction,
            )
            
            async for chunk in client.aio.models.generate_content_stream(
                model=kwargs.get("model", self.config.model),
                contents=contents,
                config=config
            ):
                if chunk.text:
                    latency_ms = (time.time() - start_time) * 1000
                    yield CompletionResponse(
                        content=chunk.text,
                        model=self.config.model,
                        provider="gemini",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        finish_reason="",
                        latency_ms=latency_ms,
                        is_complete=False
                    )
                    
        except Exception as e:
            raise ProviderError("gemini", e)
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini tokenizer."""
        try:
            client = self._get_client()
            result = await client.aio.models.count_tokens(
                model=self.config.model,
                contents=text
            )
            return result.total_tokens
        except Exception:
            return len(text) // 4
    
    async def health_check(self) -> bool:
        """Check Gemini API availability."""
        try:
            client = self._get_client()
            await client.aio.models.list()
            return True
        except Exception:
            return False
