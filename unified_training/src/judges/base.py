from abc import ABC, abstractmethod
from typing import Any, Dict, List

class Judge(ABC):
    @abstractmethod
    def score(self, inputs: List[str], outputs: List[str], gt: List[str], meta: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Returns {"score": float, "meta": {*subscores*}}
        """
        pass





