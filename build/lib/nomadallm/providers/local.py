"""
NomadaLLM Local Provider

Local LLM integration via Ollama or other local inference servers.

Security: Data never leaves the local machine - maximum privacy.
"""

import os
import time
from typing import Any, AsyncIterator, Dict, List, Optional

from nomadallm.providers.base import (
    Provider, ProviderConfig, ProviderType,
    Message, CompletionResponse
)
from nomadallm.exceptions import ProviderError


class LocalProvider(Provider):
    """Local LLM provider via Ollama.
    
    Security: All processing happens locally - no data leaves device.
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None):
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.LOCAL,
                base_url=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
                model="llama3"
            )
        super().__init__(config)
        self._client = None
    
    def _validate_config(self) -> None:
        """Validate local configuration."""
        if not self.config.base_url:
            self.config.base_url = "http://localhost:11434"
    
    def _get_client(self):
        """Get or create Ollama client."""
        if self._client is None:
            try:
                import httpx
                self._client = httpx.AsyncClient(
                    base_url=self.config.base_url,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ProviderError(
                    "local",
                    Exception("httpx package not installed. Run: pip install httpx")
                )
        return self._client
    
    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using local Ollama."""
        client = self._get_client()
        start_time = time.time()
        
        try:
            response = await client.post(
                "/api/chat",
                json={
                    "model": kwargs.get("model", self.config.model),
                    "messages": [m.to_dict() for m in messages],
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    }
                }
            )
            response.raise_for_status()
            data = response.json()
            
            latency_ms = (time.time() - start_time) * 1000
            
            return CompletionResponse(
                content=data.get("message", {}).get("content", ""),
                model=data.get("model", self.config.model),
                provider="local",
                prompt_tokens=data.get("prompt_eval_count", 0),
                completion_tokens=data.get("eval_count", 0),
                total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                finish_reason="stop",
                latency_ms=latency_ms
            )
            
        except Exception as e:
            raise ProviderError("local", e)
    
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[CompletionResponse]:
        """Stream completion using local Ollama."""
        client = self._get_client()
        start_time = time.time()
        
        try:
            async with client.stream(
                "POST",
                "/api/chat",
                json={
                    "model": kwargs.get("model", self.config.model),
                    "messages": [m.to_dict() for m in messages],
                    "stream": True,
                    "options": {
                        "temperature": kwargs.get("temperature", self.config.temperature),
                        "num_predict": kwargs.get("max_tokens", self.config.max_tokens),
                    }
                }
            ) as response:
                import json
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            latency_ms = (time.time() - start_time) * 1000
                            yield CompletionResponse(
                                content=content,
                                model=self.config.model,
                                provider="local",
                                prompt_tokens=0,
                                completion_tokens=0,
                                total_tokens=0,
                                finish_reason="" if not data.get("done") else "stop",
                                latency_ms=latency_ms,
                                is_complete=data.get("done", False)
                            )
                            
        except Exception as e:
            raise ProviderError("local", e)
    
    async def count_tokens(self, text: str) -> int:
        """Estimate token count for local model."""
        return len(text) // 4
    
    async def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available local models."""
        try:
            client = self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            client = self._get_client()
            response = await client.post(
                "/api/pull",
                json={"name": model_name}
            )
            return response.status_code == 200
        except Exception:
            return False
