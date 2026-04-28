"""
NomadaLLM Core

Cross-cutting infrastructure: protocol adapters, real-time data
ingestion, and sovereignty enforcement layers.
"""

from nomadallm.core.lsl_client import NomadaLSLReceiver

__all__ = [
    "NomadaLSLReceiver",
]
