from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import json
import torch
import numpy as np

class Logger(ABC):
    def __init__(self, log_dir: str, log_freq: int):
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.iteration_num = 0
        self.epoch_num = 0
        self.epoch_type = "train" # train | validation | test
        
        # State
        self.reset_epoch()

    def reset_epoch(self):
        self.losses: List[float] = []
        self.lrs: List[float] = []
        self.outputs: List[Any] = []
        self.gts: List[Any] = []
        self.batch_counts = 0

    def set_epoch(self, epoch_num: int, epoch_type: str):
        self.epoch_num = epoch_num
        self.epoch_type = epoch_type
        self.iteration_num = 0  # reset so iter_N = batch index within this epoch
        self.reset_epoch()

    @abstractmethod
    def log(self, **kwargs):
        """
        Log batch level metrics.
        kwargs should typically include 'loss', 'lr', 'outputs', 'gt' etc.
        """
        pass

    @abstractmethod
    def log_epoch(self):
        """
        Compute and log epoch-level metrics.
        """
        pass
        
    def _save_json(self, data: Dict[str, Any], filename: str):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, filename)
        # Append if it's a log file, or overwrite? 
        # Usually epoch logs are appended or separate files.
        # Let's write separate files per epoch or a main log file.
        # For simplicity, let's print to stdout and maybe append to a jsonl
        with open(os.path.join(self.log_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(data) + "\n")
            f.flush()






