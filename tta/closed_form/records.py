from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class PredictionRecord:
    """One causal forecast and the observations revealed after its origin."""

    origin: int
    pred_raw: torch.Tensor
    eval_target: torch.Tensor
    pogt_target: int
    observed_values: List[torch.Tensor] = field(default_factory=list)
    pred_adapted: Optional[torch.Tensor] = None
    finalized: bool = False

    @property
    def pogt_len(self) -> int:
        return len(self.observed_values)

    @property
    def gt_observed(self) -> torch.Tensor:
        if not self.observed_values:
            return self.pred_raw.new_empty((0, self.pred_raw.shape[-1]))
        return torch.stack(self.observed_values, dim=0)

    def observe(self, value: torch.Tensor) -> None:
        if self.pogt_len >= self.pred_raw.shape[0]:
            raise RuntimeError('Cannot observe beyond the forecasting horizon')
        if value.shape != self.pred_raw.shape[1:]:
            raise ValueError(
                f'Observation shape {tuple(value.shape)} does not match '
                f'{tuple(self.pred_raw.shape[1:])}'
            )
        self.observed_values.append(value.detach().clone())


@dataclass(frozen=True)
class SupervisionRecord:
    origin: int
    available_at: int
    pred_raw: torch.Tensor
    gt_observed: torch.Tensor
