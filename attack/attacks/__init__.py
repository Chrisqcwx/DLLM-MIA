import hashlib
import json

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer


class AbstractAttack(ABC):
    @abstractmethod
    def __init__(
        self,
        name: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.name = name
        self.device = device

        # Handle both DataParallel and non-DataParallel models
        if isinstance(model, torch.nn.DataParallel):
            self.device = next(model.parameters()).device
        else:
            self.device = model.device

    @abstractmethod
    def run(self, dataset: Dataset) -> Dataset:
        """Run the attack on the input data."""
        pass

    def _safe_add_column(
        self, dataset: Dataset, column_name: str, values: Sequence[Any]
    ) -> Dataset:
        """Add a column with fallback for datasets version mismatches."""
        if len(values) != len(dataset):
            raise ValueError(
                f"Column length mismatch for '{column_name}': "
                f"{len(values)} != {len(dataset)}"
            )

        try:
            return dataset.add_column(column_name, list(values))
        except TypeError as exc:
            # Some mixed datasets versions fail in DatasetInfo.copy(task_templates).
            if "task_templates" not in str(exc):
                raise

            data_dict = dataset.to_dict()
            data_dict[column_name] = list(values)

            split = getattr(dataset, "split", None)
            return Dataset.from_dict(data_dict, split=split)

    def signature(self, dataset: Dataset):
        config_str = json.dumps(self.config, sort_keys=True)
        encoded = (str(dataset.split) + self.name + config_str).encode()
        hash_obj = hashlib.sha256(encoded)
        return hash_obj.hexdigest()[:32]

    def get_base_model(self):
        return (
            self.model.module
            if isinstance(self.model, torch.nn.DataParallel)
            else self.model
        )

    def extract_features(self, batch):
        """Default feature extraction method for simple attacks"""
        # Get the attack score and reshape it as a single feature
        scores = np.array([-s for s in batch[self.name]]).reshape(-1, 1)
        return scores
