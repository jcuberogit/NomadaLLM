"""
NomadaLLM Providers

Multi-provider LLM abstraction layer supporting OpenAI, Anthropic,
Google Gemini, local models (Ollama), and embedded models (llama.cpp).

Security: Provider-agnostic interface with consistent privacy controls.
"""

from nomadallm.providers.base import Provider, ProviderConfig
from nomadallm.providers.openai import OpenAIProvider
from nomadallm.providers.anthropic import AnthropicProvider
from nomadallm.providers.gemini import GeminiProvider
from nomadallm.providers.local import LocalProvider
from nomadallm.providers.embedded import EmbeddedProvider

__all__ = [
    "Provider",
    "ProviderConfig",
    "OpenAIProvider",
    "AnthropicProvider",
    "GeminiProvider",
    "LocalProvider",
    "EmbeddedProvider",
]
