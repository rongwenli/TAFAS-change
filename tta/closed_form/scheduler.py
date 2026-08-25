from abc import ABC, abstractmethod

import torch


class POGTScheduler(ABC):
    @abstractmethod
    def get_p(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
        current_time: int,
    ) -> int:
        """Return the number of causally observed future points to wait for."""


class FixedPOGTScheduler(POGTScheduler):
    def __init__(self, pogt_len: int) -> None:
        if pogt_len <= 0:
            raise ValueError('pogt_len must be positive')
        self.pogt_len = pogt_len

    def get_p(
        self,
        x: torch.Tensor,
        pred: torch.Tensor,
        current_time: int,
    ) -> int:
        del x, pred, current_time
        return self.pogt_len
