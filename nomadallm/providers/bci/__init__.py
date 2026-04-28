"""
NomadaLLM BCI Providers

Brain-Computer Interface signal processing for sovereign neuro-AI.
All signal processing runs on-device. Raw biometric data never
leaves the privacy boundary.

Security: Biometric signatures are stripped before any data reaches the LLM.
"""

from nomadallm.providers.bci.streamer import (
    EEGStreamer,
    EEGEpoch,
    EEGFeatures,
    EpochRejected,
)

__all__ = [
    "EEGStreamer",
    "EEGEpoch",
    "EEGFeatures",
    "EpochRejected",
]
