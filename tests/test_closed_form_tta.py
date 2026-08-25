from collections import deque
import unittest

import torch

from tta.closed_form.records import PredictionRecord, SupervisionRecord
from tta.closed_form.adapter import PredictionDerivedClosedFormAdapter
from tta.closed_form.solver import ClosedFormDiagonalAdapter
from tta.closed_form.subspace import RollingPredictionSubspace, dct_basis


class ClosedFormDiagonalAdapterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.horizon = 24
        self.rank = 4
        self.n_vars = 5
        self.pogt_len = 12
        random_basis = torch.randn(self.horizon, self.rank, dtype=torch.float64)
        self.basis = torch.linalg.qr(random_basis, mode='reduced').Q
        self.solver = ClosedFormDiagonalAdapter(
            rank=self.rank,
            ridge_lambda=1e-10,
            forgetting_factor=1.0,
        )

    def test_vectorized_design_matches_matrix_expression(self):
        pred = torch.randn(self.horizon, self.n_vars, dtype=torch.float64)
        target = torch.randn(self.pogt_len, self.n_vars, dtype=torch.float64)
        coefficients = torch.randn(self.rank, dtype=torch.float64)

        design, _ = self.solver.build_system(pred, target, self.basis)
        reconstructed_design = design @ coefficients
        projected = self.basis.T @ pred
        reconstructed_matrix = (
            self.basis[: self.pogt_len]
            @ torch.diag(coefficients)
            @ projected
        ).reshape(-1)

        self.assertTrue(
            torch.allclose(
                reconstructed_design, reconstructed_matrix, atol=1e-10, rtol=1e-10
            )
        )

    def test_closed_form_recovers_known_coefficients(self):
        coefficients_true = torch.tensor(
            [0.2, -0.1, 0.05, 0.15], dtype=torch.float64
        )
        records = []
        for origin in range(3):
            pred = torch.randn(self.horizon, self.n_vars, dtype=torch.float64)
            adapted = self.solver.apply(pred, self.basis, coefficients_true)
            records.append(
                SupervisionRecord(
                    origin=origin,
                    available_at=origin + self.pogt_len,
                    pred_raw=pred,
                    gt_observed=adapted[: self.pogt_len],
                )
            )

        result = self.solver.solve(
            records, self.basis, current_time=self.pogt_len + len(records)
        )
        self.assertTrue(
            torch.allclose(
                result.coefficients,
                coefficients_true,
                atol=1e-7,
                rtol=1e-7,
            )
        )

    def test_no_residual_produces_no_correction(self):
        pred = torch.randn(self.horizon, self.n_vars, dtype=torch.float64)
        record = SupervisionRecord(
            origin=0,
            available_at=self.pogt_len,
            pred_raw=pred,
            gt_observed=pred[: self.pogt_len].clone(),
        )
        result = self.solver.solve([record], self.basis, self.pogt_len)
        adapted = self.solver.apply(pred, self.basis, result.coefficients)

        self.assertTrue(torch.allclose(result.coefficients, torch.zeros_like(result.coefficients)))
        self.assertTrue(torch.allclose(adapted, pred))

    def test_observed_prefix_is_never_changed(self):
        pred = torch.randn(self.horizon, self.n_vars, dtype=torch.float64)
        coefficients = torch.randn(self.rank, dtype=torch.float64)
        final_prediction = self.solver.replace_unobserved(
            pred, self.basis, coefficients, self.pogt_len
        )

        self.assertTrue(torch.equal(final_prediction[: self.pogt_len], pred[: self.pogt_len]))
        self.assertFalse(torch.equal(final_prediction[self.pogt_len :], pred[self.pogt_len :]))

    def test_basis_sign_flip_is_invariant(self):
        pred = torch.randn(self.horizon, self.n_vars, dtype=torch.float64)
        coefficients_true = torch.randn(self.rank, dtype=torch.float64)
        target = self.solver.apply(pred, self.basis, coefficients_true)
        signs = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float64)
        flipped_basis = self.basis * signs
        record = SupervisionRecord(
            origin=0,
            available_at=self.pogt_len,
            pred_raw=pred,
            gt_observed=target[: self.pogt_len],
        )

        result_1 = self.solver.solve([record], self.basis, self.pogt_len)
        result_2 = self.solver.solve([record], flipped_basis, self.pogt_len)
        adapted_1 = self.solver.apply(pred, self.basis, result_1.coefficients)
        adapted_2 = self.solver.apply(pred, flipped_basis, result_2.coefficients)
        self.assertTrue(torch.allclose(adapted_1, adapted_2, atol=1e-9, rtol=1e-9))

    def test_unrevealed_future_does_not_affect_solution(self):
        pred = torch.randn(self.horizon, self.n_vars, dtype=torch.float64)
        observed = torch.randn(self.pogt_len, self.n_vars, dtype=torch.float64)
        future_a = torch.randn(self.horizon - self.pogt_len, self.n_vars)
        future_b = future_a + 1000.0
        record = SupervisionRecord(
            origin=0,
            available_at=self.pogt_len,
            pred_raw=pred,
            gt_observed=observed,
        )

        result_a = self.solver.solve([record], self.basis, self.pogt_len)
        result_b = self.solver.solve([record], self.basis, self.pogt_len)
        self.assertTrue(torch.equal(result_a.coefficients, result_b.coefficients))
        self.assertFalse(torch.equal(future_a, future_b))

    def test_closed_form_operations_do_not_build_gradients(self):
        pred = torch.randn(
            self.horizon, self.n_vars, dtype=torch.float64, requires_grad=True
        )
        basis = self.basis.clone().requires_grad_(True)
        coefficients = torch.randn(self.rank, dtype=torch.float64, requires_grad=True)
        adapted = self.solver.apply(pred, basis, coefficients)
        self.assertFalse(adapted.requires_grad)

    def test_batched_apply_matches_individual_apply(self):
        predictions = torch.randn(
            6, self.horizon, self.n_vars, dtype=torch.float64
        )
        coefficients = torch.randn(self.rank, dtype=torch.float64)
        batch_result = self.solver.apply_batch(
            predictions, self.basis, coefficients
        )
        individual_result = torch.stack(
            [
                self.solver.apply(prediction, self.basis, coefficients)
                for prediction in predictions
            ],
            dim=0,
        )
        self.assertTrue(
            torch.allclose(batch_result, individual_result, atol=1e-10, rtol=1e-10)
        )

    def test_source_module_is_frozen_without_parameter_changes(self):
        model = torch.nn.Linear(4, 3)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        adapter = PredictionDerivedClosedFormAdapter.__new__(
            PredictionDerivedClosedFormAdapter
        )
        adapter.model = model
        adapter.norm_module = None
        adapter._freeze_source()
        adapter._source_parameter_versions = adapter._capture_source_parameter_versions()
        adapter._assert_source_unchanged()

        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
        self.assertTrue(
            all(torch.equal(before[name], value) for name, value in model.state_dict().items())
        )

    def test_full_target_is_evaluated_only_after_horizon_arrives(self):
        prediction = torch.randn(self.horizon, self.n_vars)
        record = PredictionRecord(
            origin=0,
            pred_raw=prediction,
            eval_target=torch.randn_like(prediction),
            pogt_target=self.pogt_len,
            pred_adapted=prediction.clone(),
            finalized=True,
        )
        adapter = PredictionDerivedClosedFormAdapter.__new__(
            PredictionDerivedClosedFormAdapter
        )
        adapter.horizon = self.horizon
        adapter.pending_evaluation = deque([record])
        adapter.source_mse = []
        adapter.source_mae = []
        adapter.adapted_mse = []
        adapter.adapted_mae = []

        adapter._evaluate_ready_predictions(self.horizon - 1)
        self.assertEqual(adapter.source_mse, [])
        self.assertEqual(len(adapter.pending_evaluation), 1)

        adapter._evaluate_ready_predictions(self.horizon)
        self.assertEqual(len(adapter.source_mse), 1)
        self.assertEqual(len(adapter.pending_evaluation), 0)


class TemporalSubspaceTest(unittest.TestCase):
    def test_dct_basis_is_orthonormal(self):
        basis = dct_basis(48, 8, dtype=torch.float64)
        self.assertTrue(
            torch.allclose(basis.T @ basis, torch.eye(8, dtype=torch.float64), atol=1e-10)
        )

    def test_rolling_pca_uses_only_predictions_already_added(self):
        estimator = RollingPredictionSubspace(
            horizon=16,
            rank=3,
            memory_size=8,
            min_pca_samples=4,
            normalization='per_trajectory',
        )
        past = torch.randn(2, 16, 3)
        future_a = torch.randn(16, 3)
        future_b = future_a + 1000.0
        for prediction in past:
            estimator.update(prediction)
        basis_before_a = estimator.estimate().basis

        estimator_copy = RollingPredictionSubspace(
            horizon=16,
            rank=3,
            memory_size=8,
            min_pca_samples=4,
            normalization='per_trajectory',
        )
        for prediction in past:
            estimator_copy.update(prediction)
        basis_before_b = estimator_copy.estimate().basis

        self.assertTrue(torch.allclose(basis_before_a, basis_before_b))
        self.assertFalse(torch.equal(future_a, future_b))


if __name__ == '__main__':
    unittest.main()
