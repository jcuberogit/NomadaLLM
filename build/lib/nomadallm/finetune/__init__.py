"""
NomadaLLM Fine-tuning Module

Local fine-tuning with LoRA adapters for dataset customization.
100% offline, no cloud required.

Security: Training data never leaves the device.
"""

from nomadallm.finetune.trainer import FineTuner
from nomadallm.finetune.datasets import DatasetLoader, DatasetFormat, Dataset

__all__ = [
    "FineTuner",
    "DatasetLoader", 
    "DatasetFormat",
]
