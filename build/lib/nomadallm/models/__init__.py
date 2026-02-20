"""
NomadaLLM Embedded Models

Pre-packaged GGUF models for offline inference.
"""

import os
from pathlib import Path

MODEL_DIR = Path(__file__).parent
DEFAULT_MODEL = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

def get_model_path(model_name: str = None) -> Path:
    """Get path to an embedded model."""
    name = model_name or DEFAULT_MODEL
    path = MODEL_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return path

def list_models() -> list:
    """List available embedded models."""
    return [f.name for f in MODEL_DIR.glob("*.gguf")]
