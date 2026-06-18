from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
from tqdm import tqdm
from ..models.base import Model
from ..loggers.base import Logger

class Epoch(ABC):
    def __init__(self, model: Model, dataloader: Any, logger: Logger, device: Optional[torch.device] = None):
        self.model = model
        # For DDP, we need the underlying module for custom methods like .generate()
        self.unwrapped_model = model.module if hasattr(model, "module") else model
        self.dataloader = dataloader
        self.logger = logger
        # If device not provided, try to infer from model
        if device is None:
            if hasattr(model, "device"):
                self.device = model.device
            elif hasattr(model, "parameters") and list(model.parameters()):
                self.device = list(model.parameters())[0].device
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

    @abstractmethod
    def run(self, **kwargs):
        pass

class TrainEpoch(Epoch):
    def __init__(self, model: Model, dataloader: Any, logger: Logger, 
                 optimizer: torch.optim.Optimizer, scheduler: Any = None, scaler: Any = None,
                 device: Optional[torch.device] = None, amp: bool = True, max_grad_norm: float = 1.0):
        super().__init__(model, dataloader, logger, device=device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.amp = amp
        self.max_grad_norm = max_grad_norm

class EvalEpoch(Epoch):
    def __init__(self, model, dataloader, logger, device=None, gen_cfg: Optional[Dict[str, Any]] = None):
        super().__init__(model, dataloader, logger, device=device)
        # Default generation config, can be overridden via constructor
        self.gen_cfg = gen_cfg or {"max_new_tokens": 5000, "num_beams": 1, "do_sample": False}
        
