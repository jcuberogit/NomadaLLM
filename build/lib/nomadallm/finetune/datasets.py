"""
NomadaLLM Dataset Loader

Load and validate datasets for fine-tuning.
Supports JSONL, CSV, and conversation formats.
"""

import json
import csv
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union
from dataclasses import dataclass, field


class DatasetFormat(Enum):
    """Supported dataset formats."""
    INSTRUCTION = "instruction"
    CONVERSATION = "conversation"
    CLASSIFICATION = "classification"
    QA = "qa"


@dataclass
class DatasetExample:
    """A single training example."""
    prompt: str
    response: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_training_format(self) -> str:
        """Convert to Llama chat format for training."""
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{self.prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{self.response}<|eot_id|>"


@dataclass
class Dataset:
    """A collection of training examples."""
    name: str
    examples: List[DatasetExample] = field(default_factory=list)
    format: DatasetFormat = DatasetFormat.INSTRUCTION
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self) -> Generator[DatasetExample, None, None]:
        yield from self.examples
    
    def add(self, example: DatasetExample) -> None:
        self.examples.append(example)
    
    def split(self, train_ratio: float = 0.9) -> tuple:
        split_idx = int(len(self.examples) * train_ratio)
        train = Dataset(name=f"{self.name}_train", examples=self.examples[:split_idx], format=self.format)
        val = Dataset(name=f"{self.name}_val", examples=self.examples[split_idx:], format=self.format)
        return train, val


class DatasetLoader:
    """Load datasets from various file formats."""
    
    @staticmethod
    def load(path: Union[str, Path], format: DatasetFormat = DatasetFormat.INSTRUCTION, name: Optional[str] = None) -> Dataset:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        
        dataset_name = name or path.stem
        
        if path.suffix.lower() == ".jsonl":
            return DatasetLoader._load_jsonl(path, format, dataset_name)
        elif path.suffix.lower() == ".csv":
            return DatasetLoader._load_csv(path, format, dataset_name)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    @staticmethod
    def _load_jsonl(path: Path, format: DatasetFormat, name: str) -> Dataset:
        dataset = Dataset(name=name, format=format)
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    example = DatasetLoader._parse_example(data, format)
                    dataset.add(example)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
        return dataset
    
    @staticmethod
    def _load_csv(path: Path, format: DatasetFormat, name: str) -> Dataset:
        dataset = Dataset(name=name, format=format)
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, 1):
                example = DatasetLoader._parse_example(dict(row), format)
                dataset.add(example)
        return dataset
    
    @staticmethod
    def _parse_example(data: Dict[str, Any], format: DatasetFormat) -> DatasetExample:
        if format == DatasetFormat.INSTRUCTION:
            instruction = data.get("instruction", "")
            input_text = data.get("input", "")
            output = data.get("output", "")
            prompt = f"{instruction}\n\nInput: {input_text}" if input_text else instruction
            return DatasetExample(prompt=prompt, response=output)
        
        elif format == DatasetFormat.CONVERSATION:
            messages = data.get("messages", [])
            user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
            assistant_msgs = [m["content"] for m in messages if m.get("role") == "assistant"]
            prompt = user_msgs[-1] if user_msgs else ""
            response = assistant_msgs[-1] if assistant_msgs else ""
            return DatasetExample(prompt=prompt, response=response)
        
        elif format == DatasetFormat.CLASSIFICATION:
            text = data.get("text", "")
            label = data.get("label", "")
            return DatasetExample(prompt=f"Classify: {text}", response=label)
        
        elif format == DatasetFormat.QA:
            question = data.get("question", "")
            answer = data.get("answer", "")
            return DatasetExample(prompt=question, response=answer)
        
        else:
            raise ValueError(f"Unknown format: {format}")
