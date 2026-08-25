from collections import deque
from dataclasses import dataclass
import math
import time
from typing import Deque, Optional

import torch


def dct_basis(
    horizon: int,
    rank: int,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the first ``rank`` orthonormal DCT-II temporal modes."""
    if horizon <= 0 or not 1 <= rank <= horizon:
        raise ValueError(f'Expected 1 <= rank <= horizon, got {rank} and {horizon}')

    positions = torch.arange(horizon, device=device, dtype=dtype).unsqueeze(1)
    frequencies = torch.arange(rank, device=device, dtype=dtype).unsqueeze(0)
    basis = torch.cos(math.pi / horizon * (positions + 0.5) * frequencies)
    basis[:, 0] *= math.sqrt(1.0 / horizon)
    if rank > 1:
        basis[:, 1:] *= math.sqrt(2.0 / horizon)
    return basis


@dataclass(frozen=True)
class SubspaceResult:
    basis: torch.Tensor
    source: str
    explained_variance_ratio: float
    elapsed_seconds: float


class RollingPredictionSubspace:
    """Causal rolling PCA over raw source-model forecast trajectories."""

    def __init__(
        self,
        horizon: int,
        rank: int,
        memory_size: int,
        min_pca_samples: int,
        normalization: str = 'per_trajectory',
        eps: float = 1e-6,
    ) -> None:
        if memory_size <= 0:
            raise ValueError('memory_size must be positive')
        if min_pca_samples <= 0:
            raise ValueError('min_pca_samples must be positive')
        if normalization not in ('none', 'per_trajectory'):
            raise ValueError(f'Unsupported PCA normalization: {normalization}')
        if not 1 <= rank <= horizon:
            raise ValueError(f'Expected 1 <= rank <= horizon, got {rank} and {horizon}')

        self.horizon = horizon
        self.rank = rank
        self.memory_size = memory_size
        self.min_pca_samples = min_pca_samples
        self.normalization = normalization
        self.eps = eps
        self.memory: Deque[torch.Tensor] = deque(maxlen=memory_size)

    def reset(self) -> None:
        self.memory.clear()

    def update(self, pred_raw: torch.Tensor) -> None:
        if pred_raw.ndim != 2 or pred_raw.shape[0] != self.horizon:
            raise ValueError(
                f'Expected prediction [H, C] with H={self.horizon}, '
                f'got {tuple(pred_raw.shape)}'
            )
        self.memory.append(pred_raw.detach())

    @property
    def num_predictions(self) -> int:
        return len(self.memory)

    @property
    def num_trajectories(self) -> int:
        if not self.memory:
            return 0
        return len(self.memory) * self.memory[0].shape[1]

    @torch.no_grad()
    def estimate(self) -> SubspaceResult:
        if not self.memory:
            raise RuntimeError('At least one prediction is required to estimate a subspace')

        start = time.perf_counter()
        reference = self.memory[-1]
        enough_samples = self.num_trajectories >= max(
            self.min_pca_samples, self.rank
        )
        if not enough_samples:
            basis = dct_basis(
                self.horizon,
                self.rank,
                device=reference.device,
                dtype=reference.dtype,
            )
            return SubspaceResult(
                basis=basis,
                source='DCT',
                explained_variance_ratio=0.0,
                elapsed_seconds=time.perf_counter() - start,
            )

        predictions = torch.stack(tuple(self.memory), dim=0)
        trajectories = predictions.permute(0, 2, 1).reshape(-1, self.horizon)

        if self.normalization == 'per_trajectory':
            trajectory_mean = trajectories.mean(dim=1, keepdim=True)
            trajectory_var = trajectories.var(dim=1, keepdim=True, unbiased=False)
            trajectories = (trajectories - trajectory_mean) / torch.sqrt(
                trajectory_var + self.eps
            )

        centered = trajectories - trajectories.mean(dim=0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        basis = vh[: self.rank].transpose(0, 1).contiguous()
        spectrum = singular_values.square()
        explained = (
            spectrum[: self.rank].sum() / spectrum.sum().clamp_min(self.eps)
        ).item()

        return SubspaceResult(
            basis=basis,
            source='PCA',
            explained_variance_ratio=float(explained),
            elapsed_seconds=time.perf_counter() - start,
        )
