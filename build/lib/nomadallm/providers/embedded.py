"""
NomadaLLM Embedded Provider

True embedded LLM using llama-cpp-python.
No external servers, no network calls, 100% on-device.

Security: Model runs in-process. Data never leaves the application.
"""

import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from nomadallm.providers.base import (
    Provider, ProviderConfig, ProviderType,
    Message, CompletionResponse
)
from nomadallm.exceptions import ProviderError


# Default model configuration
DEFAULT_MODEL_NAME = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL_URL = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL_SIZE_MB = 600


class EmbeddedProvider(Provider):
    """Embedded LLM provider using llama-cpp-python.
    
    Security: All processing happens in-process - maximum privacy.
    No external servers, no network calls after model download.
    
    Usage:
        provider = EmbeddedProvider()
        response = await provider.complete([Message(role="user", content="Hello")])
    """
    
    def __init__(self, config: Optional[ProviderConfig] = None, model_path: Optional[str] = None):
        """Initialize embedded provider.
        
        Args:
            config: Provider configuration
            model_path: Path to GGUF model file. If None, uses default location.
        """
        if config is None:
            config = ProviderConfig(
                provider_type=ProviderType.LOCAL,
                model="llama-3.2-1b"
            )
        super().__init__(config)
        
        self._llm = None
        self._model_path = model_path or self._get_default_model_path()
    
    def _validate_config(self) -> None:
        """Validate embedded configuration."""
        pass
    
    def _get_default_model_path(self) -> str:
        """Get default model path - first check bundled, then user directory."""
        # First, check if model is bundled with the package
        bundled_path = Path(__file__).parent.parent / "models" / DEFAULT_MODEL_NAME
        if bundled_path.exists():
            return str(bundled_path)
        
        # Fallback to user's home directory
        home = Path.home()
        nomada_dir = home / ".nomadallm" / "models"
        nomada_dir.mkdir(parents=True, exist_ok=True)
        return str(nomada_dir / DEFAULT_MODEL_NAME)
    
    def _ensure_model_exists(self) -> str:
        """Ensure model file exists, download if needed.
        
        Returns:
            Path to the model file.
        """
        model_path = Path(self._model_path)
        
        if model_path.exists():
            return str(model_path)
        
        # Model doesn't exist, need to download
        print(f"[NomadaLLM] Model not found at {model_path}")
        print(f"[NomadaLLM] Downloading {DEFAULT_MODEL_NAME} ({DEFAULT_MODEL_SIZE_MB}MB)...")
        
        self._download_model(str(model_path))
        
        return str(model_path)
    
    def _download_model(self, destination: str) -> None:
        """Download the default model.
        
        Args:
            destination: Where to save the model file.
        """
        import urllib.request
        import shutil
        
        try:
            # Create parent directory if needed
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            def progress_hook(count, block_size, total_size):
                percent = int(count * block_size * 100 / total_size)
                print(f"\r[NomadaLLM] Downloading: {percent}%", end="", flush=True)
            
            urllib.request.urlretrieve(
                DEFAULT_MODEL_URL,
                destination,
                reporthook=progress_hook
            )
            print(f"\n[NomadaLLM] Download complete: {destination}")
            
        except Exception as e:
            raise ProviderError(
                "embedded",
                Exception(f"Failed to download model: {e}")
            )
    
    def _get_llm(self):
        """Get or create the LLM instance."""
        if self._llm is None:
            try:
                from llama_cpp import Llama
            except ImportError:
                raise ProviderError(
                    "embedded",
                    Exception(
                        "llama-cpp-python not installed. Run: pip install llama-cpp-python"
                    )
                )
            
            # Ensure model exists
            model_path = self._ensure_model_exists()
            
            # Initialize llama.cpp
            print(f"[NomadaLLM] Loading model: {model_path}")
            start_time = time.time()
            
            self._llm = Llama(
                model_path=model_path,
                n_ctx=4096,              # Context window
                n_threads=4,             # CPU threads
                n_gpu_layers=0,          # CPU only by default (change for GPU)
                verbose=False,           # Quiet mode
                chat_format="llama-3"    # Llama 3 chat template
            )
            
            load_time = time.time() - start_time
            print(f"[NomadaLLM] Model loaded in {load_time:.2f}s")
        
        return self._llm
    
    async def complete(
        self,
        messages: List[Message],
        **kwargs
    ) -> CompletionResponse:
        """Generate completion using embedded LLM.
        
        Security: All processing happens locally.
        """
        llm = self._get_llm()
        start_time = time.time()
        
        try:
            # Convert messages to llama.cpp format
            llama_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            
            # Generate completion
            response = llm.create_chat_completion(
                messages=llama_messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens or 512),
                temperature=kwargs.get("temperature", self.config.temperature or 0.7),
                top_p=kwargs.get("top_p", 0.9),
                stop=kwargs.get("stop", None),
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract response
            content = response["choices"][0]["message"]["content"]
            usage = response.get("usage", {})
            
            return CompletionResponse(
                content=content,
                model="llama-3.2-1b-instruct",
                provider="embedded",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                finish_reason=response["choices"][0].get("finish_reason", "stop"),
                latency_ms=latency_ms
            )
            
        except Exception as e:
            raise ProviderError("embedded", e)
    
    async def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> AsyncIterator[CompletionResponse]:
        """Stream completion using embedded LLM.
        
        Security: All processing happens locally.
        """
        llm = self._get_llm()
        start_time = time.time()
        
        try:
            # Convert messages to llama.cpp format
            llama_messages = [
                {"role": m.role, "content": m.content}
                for m in messages
            ]
            
            # Stream completion
            stream = llm.create_chat_completion(
                messages=llama_messages,
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens or 512),
                temperature=kwargs.get("temperature", self.config.temperature or 0.7),
                stream=True,
            )
            
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                
                if content:
                    latency_ms = (time.time() - start_time) * 1000
                    finish_reason = chunk["choices"][0].get("finish_reason")
                    
                    yield CompletionResponse(
                        content=content,
                        model="llama-3.2-1b-instruct",
                        provider="embedded",
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                        finish_reason=finish_reason or "",
                        latency_ms=latency_ms,
                        is_complete=finish_reason is not None
                    )
                    
        except Exception as e:
            raise ProviderError("embedded", e)
    
    async def count_tokens(self, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        llm = self._get_llm()
        tokens = llm.tokenize(text.encode("utf-8"))
        return len(tokens)
    
    async def health_check(self) -> bool:
        """Check if model is loaded and ready."""
        try:
            self._get_llm()
            return True
        except Exception:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "name": "Llama-3.2-1B-Instruct",
            "version": "Q4_K_M",
            "size_mb": DEFAULT_MODEL_SIZE_MB,
            "path": self._model_path,
            "loaded": self._llm is not None,
            "context_length": 4096,
            "provider": "embedded"
        }
    
    def unload(self) -> None:
        """Unload the model from memory."""
        if self._llm is not None:
            del self._llm
            self._llm = None
            print("[NomadaLLM] Model unloaded")
