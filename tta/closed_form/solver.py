from dataclasses import dataclass
import time
from collections import defaultdict
from typing import DefaultDict, Iterable, List, Tuple

import torch

from tta.closed_form.records import SupervisionRecord


@dataclass(frozen=True)
class SolveResult:
    coefficients: torch.Tensor
    condition_number: float
    elapsed_seconds: float
    num_supervision: int


class ClosedFormDiagonalAdapter:
    """Mode-wise ridge calibration without gradients or an optimizer."""

    def __init__(
        self,
        rank: int,
        ridge_lambda: float,
        forgetting_factor: float,
    ) -> None:
        if rank <= 0:
            raise ValueError('rank must be positive')
        if ridge_lambda <= 0:
            raise ValueError('ridge_lambda must be positive')
        if not 0.0 <= forgetting_factor <= 1.0:
            raise ValueError('forgetting_factor must be in [0, 1]')
        self.rank = rank
        self.ridge_lambda = ridge_lambda
        self.forgetting_factor = forgetting_factor

    @torch.no_grad()
    def build_system(
        self,
        pred_raw: torch.Tensor,
        gt_observed: torch.Tensor,
        basis: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pred_raw.ndim != 2 or gt_observed.ndim != 2 or basis.ndim != 2:
            raise ValueError('pred_raw, gt_observed, and basis must all be matrices')
        horizon, n_vars = pred_raw.shape
        p = gt_observed.shape[0]
        if not 0 < p <= horizon:
            raise ValueError(f'POGT length must be in [1, {horizon}], got {p}')
        if gt_observed.shape[1] != n_vars:
            raise ValueError('Prediction and observation variable counts do not match')
        if basis.shape != (horizon, self.rank):
            raise ValueError(
                f'Expected basis {(horizon, self.rank)}, got {tuple(basis.shape)}'
            )

        basis_observed = basis[:p]
        projected = basis.transpose(0, 1) @ pred_raw
        basis_corrections = basis_observed.unsqueeze(-1) * projected.unsqueeze(0)
        design = basis_corrections.permute(0, 2, 1).reshape(p * n_vars, self.rank)
        residual = (gt_observed - pred_raw[:p]).reshape(p * n_vars)
        return design, residual

    @torch.no_grad()
    def solve(
        self,
        records: Iterable[SupervisionRecord],
        basis: torch.Tensor,
        current_time: int,
    ) -> SolveResult:
        records = list(records)
        start = time.perf_counter()
        solve_dtype = torch.float64
        statistics = torch.zeros(
            (self.rank, self.rank), device=basis.device, dtype=solve_dtype
        )
        right_hand_side = torch.zeros(
            self.rank, device=basis.device, dtype=solve_dtype
        )

        records_by_pogt_len: DefaultDict[int, List[SupervisionRecord]] = defaultdict(list)
        for record in records:
            records_by_pogt_len[record.gt_observed.shape[0]].append(record)

        basis_solve = basis.to(solve_dtype)
        for pogt_len, record_group in records_by_pogt_len.items():
            predictions = torch.stack(
                [record.pred_raw for record in record_group], dim=0
            ).to(solve_dtype)
            observations = torch.stack(
                [record.gt_observed for record in record_group], dim=0
            ).to(solve_dtype)
            projected = torch.einsum('hr,bhc->brc', basis_solve, predictions)
            basis_corrections = (
                basis_solve[:pogt_len].unsqueeze(0).unsqueeze(-1)
                * projected.unsqueeze(1)
            )
            batch_size, _, _, n_vars = basis_corrections.shape
            design = basis_corrections.permute(0, 1, 3, 2).reshape(
                batch_size, pogt_len * n_vars, self.rank
            )
            residual = (observations - predictions[:, :pogt_len]).reshape(
                batch_size, pogt_len * n_vars
            )
            weights = torch.tensor(
                [
                    self.forgetting_factor
                    ** max(0, current_time - record.available_at)
                    for record in record_group
                ],
                device=basis.device,
                dtype=solve_dtype,
            )
            statistics.add_(
                torch.einsum('bnr,bns,b->rs', design, design, weights)
            )
            right_hand_side.add_(
                torch.einsum('bnr,bn,b->r', design, residual, weights)
            )

        identity = torch.eye(self.rank, device=basis.device, dtype=solve_dtype)
        system = statistics + self.ridge_lambda * identity
        coefficients = torch.linalg.solve(system, right_hand_side)
        if not torch.isfinite(coefficients).all():
            raise RuntimeError('Closed-form solve produced non-finite coefficients')
        condition_number = torch.linalg.cond(system).item()
        return SolveResult(
            coefficients=coefficients.to(dtype=basis.dtype),
            condition_number=float(condition_number),
            elapsed_seconds=time.perf_counter() - start,
            num_supervision=len(records),
        )

    @torch.no_grad()
    def apply(
        self,
        pred_raw: torch.Tensor,
        basis: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        projected = basis.transpose(0, 1) @ pred_raw
        correction = basis @ (coefficients.unsqueeze(1) * projected)
        return pred_raw + correction

    @torch.no_grad()
    def apply_batch(
        self,
        predictions: torch.Tensor,
        basis: torch.Tensor,
        coefficients: torch.Tensor,
    ) -> torch.Tensor:
        if predictions.ndim != 3:
            raise ValueError('Expected batched predictions with shape [B, H, C]')
        projected = torch.einsum('hr,bhc->brc', basis, predictions)
        correction = torch.einsum(
            'hr,brc->bhc', basis, coefficients.view(1, -1, 1) * projected
        )
        return predictions + correction

    @torch.no_grad()
    def replace_unobserved(
        self,
        pred_raw: torch.Tensor,
        basis: torch.Tensor,
        coefficients: torch.Tensor,
        pogt_len: int,
    ) -> torch.Tensor:
        if not 0 <= pogt_len <= pred_raw.shape[0]:
            raise ValueError('Invalid observed-prefix length')
        fully_adapted = self.apply(pred_raw, basis, coefficients)
        final_prediction = pred_raw.clone()
        final_prediction[pogt_len:] = fully_adapted[pogt_len:]
        return final_prediction
