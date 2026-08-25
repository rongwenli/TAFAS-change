from collections import deque
import json
import os
import time
from typing import Deque, Optional

import numpy as np
import torch

from datasets.loader import get_test_dataloader
from models.forecast import forecast
from tta.closed_form.records import PredictionRecord, SupervisionRecord
from tta.closed_form.scheduler import FixedPOGTScheduler
from tta.closed_form.solver import ClosedFormDiagonalAdapter, SolveResult
from tta.closed_form.subspace import RollingPredictionSubspace, dct_basis
from utils.misc import mkdir


class PredictionDerivedClosedFormAdapter:
    """Causal prediction-subspace TTA with an analytical ridge update."""

    def __init__(self, cfg, model: torch.nn.Module, norm_module=None) -> None:
        self.cfg = cfg
        self.model = model
        self.norm_module = norm_module
        self.method_cfg = cfg.TTA.CLOSED_FORM
        self.horizon = cfg.DATA.PRED_LEN
        self.rank = self.method_cfg.RANK
        self.pogt_len = self.method_cfg.POGT_LEN

        if not 1 <= self.rank <= self.horizon:
            raise ValueError('CLOSED_FORM.RANK must be in [1, DATA.PRED_LEN]')
        if not 1 <= self.pogt_len < self.horizon:
            raise ValueError('CLOSED_FORM.POGT_LEN must be in [1, DATA.PRED_LEN - 1]')
        if self.method_cfg.SUBSPACE_UPDATE_INTERVAL <= 0:
            raise ValueError('CLOSED_FORM.SUBSPACE_UPDATE_INTERVAL must be positive')
        if self.method_cfg.SUPERVISION_BUFFER_SIZE <= 0:
            raise ValueError('CLOSED_FORM.SUPERVISION_BUFFER_SIZE must be positive')

        self._freeze_source()
        self._source_parameter_versions = self._capture_source_parameter_versions()
        self.test_loader = get_test_dataloader(cfg)
        if cfg.TEST.SHUFFLE:
            raise ValueError('Closed-form TTA requires TEST.SHUFFLE=False')

        self.device = self._model_device()
        self.subspace = RollingPredictionSubspace(
            horizon=self.horizon,
            rank=self.rank,
            memory_size=self.method_cfg.PREDICTION_MEMORY_SIZE,
            min_pca_samples=self.method_cfg.MIN_PCA_SAMPLES,
            normalization=self.method_cfg.PCA_NORMALIZATION,
        )
        self.solver = ClosedFormDiagonalAdapter(
            rank=self.rank,
            ridge_lambda=self.method_cfg.RIDGE_LAMBDA,
            forgetting_factor=self.method_cfg.FORGETTING_FACTOR,
        )
        self.scheduler = FixedPOGTScheduler(self.pogt_len)
        self.basis = dct_basis(
            self.horizon, self.rank, device=self.device, dtype=torch.float32
        )
        self.coefficients = torch.zeros(
            self.rank, device=self.device, dtype=torch.float32
        )
        self.active_records: Deque[PredictionRecord] = deque()
        self.pending_evaluation: Deque[PredictionRecord] = deque()
        self.supervision_buffer: Deque[SupervisionRecord] = deque(
            maxlen=self.method_cfg.SUPERVISION_BUFFER_SIZE
        )

        self.source_mse = []
        self.source_mae = []
        self.adapted_mse = []
        self.adapted_mae = []
        self.forward_time = 0.0
        self.subspace_update_time = 0.0
        self.closed_form_solve_time = 0.0
        self.total_tta_time = 0.0
        self.peak_memory_bytes = 0
        self.num_adaptations = 0
        self.num_subspace_updates = 0
        self.subspace_source = 'DCT'
        self.explained_variance_ratio = 0.0
        self.last_solve: Optional[SolveResult] = None
        self.num_forecasts = 0

    def _model_device(self) -> torch.device:
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            return parameter.device
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _freeze_source(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if self.norm_module is not None:
            self.norm_module.eval()
            for parameter in self.norm_module.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None

    def _capture_source_parameter_versions(self):
        versions = {
            f'model.{name}': parameter._version
            for name, parameter in self.model.named_parameters()
        }
        if self.norm_module is not None:
            versions.update(
                {
                    f'norm_module.{name}': parameter._version
                    for name, parameter in self.norm_module.named_parameters()
                }
            )
        return versions

    def _assert_source_unchanged(self) -> None:
        current_versions = self._capture_source_parameter_versions()
        if current_versions != self._source_parameter_versions:
            raise RuntimeError('The frozen source model was modified during closed-form TTA')
        modules = [self.model]
        if self.norm_module is not None:
            modules.append(self.norm_module)
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad or parameter.grad is not None:
                    raise RuntimeError('The source model accumulated test-time gradients')

    @staticmethod
    def _cuda_sync() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _update_subspace_if_due(self) -> None:
        interval = self.method_cfg.SUBSPACE_UPDATE_INTERVAL
        if self.num_forecasts % interval != 0:
            return

        self._cuda_sync()
        result = self.subspace.estimate()
        self._cuda_sync()
        self.basis = result.basis
        self.subspace_source = result.source
        self.explained_variance_ratio = result.explained_variance_ratio
        self.subspace_update_time += result.elapsed_seconds
        self.num_subspace_updates += 1

    def _add_forecast(
        self,
        origin: int,
        pred_raw: torch.Tensor,
        eval_target: torch.Tensor,
    ) -> None:
        pogt_target = self.scheduler.get_p(
            x=pred_raw.new_empty(0), pred=pred_raw, current_time=origin
        )
        if not 1 <= pogt_target < self.horizon:
            raise ValueError('The POGT scheduler returned an invalid prefix length')
        record = PredictionRecord(
            origin=origin,
            pred_raw=pred_raw.detach().clone(),
            eval_target=eval_target.detach().clone(),
            pogt_target=pogt_target,
        )
        self.active_records.append(record)
        self.subspace.update(record.pred_raw)
        self.num_forecasts += 1
        self._update_subspace_if_due()

    def _solve_and_recalculate(self, current_time: int) -> None:
        self._cuda_sync()
        solve_result = self.solver.solve(
            self.supervision_buffer, self.basis, current_time
        )
        self._cuda_sync()
        self.coefficients = solve_result.coefficients
        self.last_solve = solve_result
        self.closed_form_solve_time += solve_result.elapsed_seconds
        self.num_adaptations += 1

        if self.method_cfg.RECALCULATE_ACTIVE:
            records = list(self.active_records)
            predictions = torch.stack([record.pred_raw for record in records], dim=0)
            adapted_batch = self.solver.apply_batch(
                predictions, self.basis, self.coefficients
            )
            for record, fully_adapted in zip(records, adapted_batch):
                record.pred_adapted = record.pred_raw.clone()
                record.pred_adapted[record.pogt_len :] = fully_adapted[
                    record.pogt_len :
                ]

        log_interval = self.method_cfg.LOG_INTERVAL
        if log_interval > 0 and self.num_adaptations % log_interval == 0:
            print('[PredictionDerivedClosedFormTTA]')
            print(f'time                 = {current_time}')
            print(f'num_pred_memory      = {self.subspace.num_predictions}')
            print(f'num_supervision      = {len(self.supervision_buffer)}')
            print(f'subspace_source      = {self.subspace_source}')
            print(f'rank                 = {self.rank}')
            print(f'POGT lengths         = {[r.gt_observed.shape[0] for r in self.supervision_buffer]}')
            print(f'ridge_lambda         = {self.method_cfg.RIDGE_LAMBDA}')
            print(f'condition_number     = {solve_result.condition_number:.4e}')
            print(f'||a||_2              = {self.coefficients.norm().item():.4e}')
            print(f'solve_time_ms        = {solve_result.elapsed_seconds * 1000:.3f}')

    def _finalize_prediction(self, record: PredictionRecord) -> None:
        if record.pred_adapted is None:
            record.pred_adapted = self.solver.replace_unobserved(
                record.pred_raw,
                self.basis,
                self.coefficients,
                record.pogt_len,
            )
        if not torch.equal(
            record.pred_adapted[: record.pogt_len],
            record.pred_raw[: record.pogt_len],
        ):
            raise RuntimeError('Anti-leakage invariant violated: observed prefix changed')

        record.finalized = True
        self.pending_evaluation.append(record)

    def _evaluate_ready_predictions(self, current_time: int) -> None:
        while (
            self.pending_evaluation
            and current_time - self.pending_evaluation[0].origin >= self.horizon
        ):
            record = self.pending_evaluation.popleft()
            if record.pred_adapted is None:
                raise RuntimeError('A forecast reached evaluation before being finalized')

            # The full target is read only after its complete horizon has occurred.
            source_error = record.pred_raw - record.eval_target
            adapted_error = record.pred_adapted - record.eval_target
            self.source_mse.append(source_error.square().mean().item())
            self.source_mae.append(source_error.abs().mean().item())
            self.adapted_mse.append(adapted_error.square().mean().item())
            self.adapted_mae.append(adapted_error.abs().mean().item())

    def _advance_time(self, current_time: int) -> None:
        matured = []
        for record in self.active_records:
            if record.origin >= current_time or record.pogt_len >= record.pogt_target:
                continue
            expected_step = current_time - record.origin - 1
            if expected_step != record.pogt_len:
                raise RuntimeError('Non-sequential POGT arrival detected')
            record.observe(record.eval_target[expected_step])
            if record.pogt_len == record.pogt_target:
                matured.append(record)

        if matured:
            for record in matured:
                self.supervision_buffer.append(
                    SupervisionRecord(
                        origin=record.origin,
                        available_at=current_time,
                        pred_raw=record.pred_raw,
                        gt_observed=record.gt_observed,
                    )
                )
            self._solve_and_recalculate(current_time)
        if matured and not self.method_cfg.RECALCULATE_ACTIVE:
            for record in matured:
                record.pred_adapted = self.solver.replace_unobserved(
                    record.pred_raw,
                    self.basis,
                    self.coefficients,
                    record.pogt_len,
                )
        if matured:
            for record in matured:
                self._finalize_prediction(record)
            self.active_records = deque(
                record for record in self.active_records if not record.finalized
            )
        self._evaluate_ready_predictions(current_time)

    def _save_results(self) -> None:
        result_dir = mkdir(self.cfg.RESULT_DIR)
        arrays = {
            'source_mse': np.asarray(self.source_mse),
            'source_mae': np.asarray(self.source_mae),
            'adapted_mse': np.asarray(self.adapted_mse),
            'adapted_mae': np.asarray(self.adapted_mae),
        }
        for name, values in arrays.items():
            np.save(os.path.join(result_dir, f'{name}.npy'), values)
        np.save(
            os.path.join(result_dir, 'coefficients.npy'),
            self.coefficients.detach().cpu().numpy(),
        )

        summary = {
            'source_mse': float(arrays['source_mse'].mean()),
            'source_mae': float(arrays['source_mae'].mean()),
            'adapted_mse': float(arrays['adapted_mse'].mean()),
            'adapted_mae': float(arrays['adapted_mae'].mean()),
            'num_forecasts': self.num_forecasts,
            'num_adaptations': self.num_adaptations,
            'num_subspace_updates': self.num_subspace_updates,
            'subspace_source': self.subspace_source,
            'explained_variance_ratio': self.explained_variance_ratio,
            'rank': self.rank,
            'pogt_len': self.pogt_len,
            'prediction_memory_size': self.method_cfg.PREDICTION_MEMORY_SIZE,
            'supervision_buffer_size': self.method_cfg.SUPERVISION_BUFFER_SIZE,
            'ridge_lambda': self.method_cfg.RIDGE_LAMBDA,
            'forgetting_factor': self.method_cfg.FORGETTING_FACTOR,
            'coefficient_norm': float(self.coefficients.norm().item()),
            'last_condition_number': (
                self.last_solve.condition_number if self.last_solve else None
            ),
            'forward_time_seconds': self.forward_time,
            'subspace_update_time_seconds': self.subspace_update_time,
            'closed_form_solve_time_seconds': self.closed_form_solve_time,
            'total_tta_time_seconds': self.total_tta_time,
            'peak_memory_bytes': self.peak_memory_bytes,
        }
        with open(os.path.join(result_dir, 'closed_form_tta.json'), 'w') as handle:
            json.dump(summary, handle, indent=2)

        print('After Prediction-Derived Closed-Form TTA')
        print(
            f"Source  MSE: {summary['source_mse']:.4f}, "
            f"MAE: {summary['source_mae']:.4f}"
        )
        print(
            f"Adapted MSE: {summary['adapted_mse']:.4f}, "
            f"MAE: {summary['adapted_mae']:.4f}"
        )
        print(
            f"Adaptations: {self.num_adaptations}, "
            f"PCA updates: {self.num_subspace_updates}, "
            f"solve time: {self.closed_form_solve_time:.3f}s"
        )
        print()

    @torch.no_grad()
    def adapt(self) -> None:
        self._freeze_source()
        start_allocated = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            start_allocated = torch.cuda.memory_allocated()
        total_start = time.perf_counter()

        origin = 0
        for inputs in self.test_loader:
            self._cuda_sync()
            forward_start = time.perf_counter()
            pred_batch, target_batch = forecast(
                self.cfg, inputs, self.model, self.norm_module
            )
            self._cuda_sync()
            self.forward_time += time.perf_counter() - forward_start

            for pred_raw, eval_target in zip(pred_batch, target_batch):
                self._add_forecast(origin, pred_raw, eval_target)
                self._advance_time(origin)
                origin += 1

        for current_time in range(origin, origin + self.horizon):
            self._advance_time(current_time)

        if self.active_records:
            raise RuntimeError(
                f'{len(self.active_records)} forecasts were not causally finalized'
            )
        if self.pending_evaluation:
            raise RuntimeError(
                f'{len(self.pending_evaluation)} forecasts were evaluated too early'
            )
        if len(self.adapted_mse) != len(self.test_loader.dataset):
            raise RuntimeError('Metric count does not match the test dataset length')
        self._assert_source_unchanged()

        self._cuda_sync()
        self.total_tta_time = time.perf_counter() - total_start
        if torch.cuda.is_available():
            self.peak_memory_bytes = max(
                0, torch.cuda.max_memory_allocated() - start_allocated
            )
        self._save_results()
        self.model.eval()
