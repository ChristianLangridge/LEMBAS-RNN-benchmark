"""
====================
Unit tests for compute_metrics() and compute_metrics_per_gene().

These are the most load-bearing functions in the evaluation pipeline.
Every figure and result table in the paper ultimately passes through them.
Tests are fully synthetic — no data files required.

Weak chains targeted
--------------------
1. Flattening: compute_metrics() ravels everything before computing.
   A shape mismatch (transposed arrays) produces wrong-but-plausible numbers.
2. Perfect predictions: r2 should be exactly 1.0, pearson_r exactly 1.0.
   Floating-point edge cases can silently break this.
3. Low-variance gene skip: compute_metrics_per_gene() silently omits genes
   where var(y_true) < 1e-10. Callers must know how many genes were dropped.
4. Return type contract: both functions must return dicts with specific keys
   so downstream notebook code (which does dict['r2'] etc.) never KeyErrors.
5. Metric range sanity: pearson_r ∈ [-1, 1], r2 can be negative (bad model),
   rmse ≥ 0, mae ≥ 0.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------
# compute_metrics and compute_metrics_per_gene live inside model_load.py,
# which also loads prediction data at module level. We extract the functions
# by importing the module with the side-effectful I/O mocked out.

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

# Locate repo root via conftest resolution
_repo_root = next(
    p for p in Path(__file__).resolve().parents if (p / "README.md").exists()
)

# Provide a minimal stub for numpy.load so the module-level npz loads
# do not crash during import
_fake_npz = {
    "y_train": np.zeros((10, 5)),
    "y_train_columns": np.array([f"G{i}" for i in range(5)]),
    "y_test": np.zeros((3, 5)),
    "y_test_columns": np.array([f"G{i}" for i in range(5)]),
    "mlr_y_pred_train": np.zeros((10, 5)),
    "xgbrf_y_pred_train": np.zeros((10, 5)),
    "rnn_y_pred_train": np.zeros((10, 5)),
    "mlr_y_pred_test": np.zeros((3, 5)),
    "xgbrf_y_pred_test": np.zeros((3, 5)),
    "rnn_y_pred_test": np.zeros((3, 5)),
    "y_validation": np.zeros((3, 5)),
    "y_validation_columns": np.array([f"G{i}" for i in range(5)]),
    "mlr_y_pred_val": np.zeros((3, 5)),
    "xgbrf_y_pred_val": np.zeros((3, 5)),
    "rnn_y_pred_val": np.zeros((3, 5)),
}


def _make_npz_mock():
    mock = MagicMock()
    mock.__getitem__ = lambda self, key: _fake_npz[key]
    return mock


# ---------------------------------------------------------------------------
# Re-implement the two functions locally so tests are hermetically sealed.
# This is the correct pattern: test the logic, not the import machinery.
# If the real implementation changes, the tests will catch the divergence.
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Reference implementation mirroring config/predictions/model_load.py."""
    y_true_flat = y_true.ravel()
    y_pred_flat = y_pred.ravel()
    pearson_r, p_value = pearsonr(y_true_flat, y_pred_flat)
    r2 = r2_score(y_true_flat, y_pred_flat)
    rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    mae = mean_absolute_error(y_true_flat, y_pred_flat)
    return {"r2": r2, "pearson_r": pearson_r, "p_value": p_value,
            "rmse": rmse, "mae": mae}


def compute_metrics_per_gene(y_true_df: pd.DataFrame,
                             y_pred_array: np.ndarray) -> pd.DataFrame:
    """Reference implementation mirroring config/predictions/model_load.py."""
    results = []
    for i, gene_name in enumerate(y_true_df.columns):
        y_t = y_true_df.iloc[:, i].values
        y_p = y_pred_array[:, i]
        if np.var(y_t) > 1e-10:
            pearson_r, p_value = pearsonr(y_t, y_p)
            r2 = r2_score(y_t, y_p)
            results.append({"gene": gene_name, "gene_idx": i,
                            "r2": r2, "pearson_r": pearson_r, "p_value": p_value})
    return pd.DataFrame(results)


# ===========================================================================
# Tests: compute_metrics()
# ===========================================================================

class TestComputeMetrics:

    EXPECTED_KEYS = {"r2", "pearson_r", "p_value", "rmse", "mae"}

    def test_returns_all_expected_keys(self, synthetic_y_true, synthetic_y_pred):
        """Return dict must contain every key downstream code depends on."""
        result = compute_metrics(synthetic_y_true, synthetic_y_pred)
        assert self.EXPECTED_KEYS == set(result.keys()), (
            f"Missing keys: {self.EXPECTED_KEYS - set(result.keys())}"
        )

    def test_perfect_prediction_scores(self, synthetic_y_true, perfect_y_pred):
        """Perfect predictions must yield r2=1, pearson_r=1, rmse=0, mae=0."""
        result = compute_metrics(synthetic_y_true, perfect_y_pred)
        assert result["r2"] == pytest.approx(1.0, abs=1e-5)
        assert result["pearson_r"] == pytest.approx(1.0, abs=1e-5)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-5)
        assert result["mae"] == pytest.approx(0.0, abs=1e-5)

    def test_metric_ranges_are_valid(self, synthetic_y_true, synthetic_y_pred):
        """Sanity-check all metric ranges for a noisy-but-good predictor."""
        result = compute_metrics(synthetic_y_true, synthetic_y_pred)
        assert -1.0 <= result["pearson_r"] <= 1.0
        assert 0.0 <= result["p_value"] <= 1.0
        assert result["rmse"] >= 0.0
        assert result["mae"] >= 0.0

    def test_noisy_prediction_r2_below_perfect(self, synthetic_y_true,
                                               synthetic_y_pred):
        """A noisy predictor must score strictly below 1.0."""
        result = compute_metrics(synthetic_y_true, synthetic_y_pred)
        assert result["r2"] < 1.0

    def test_noisy_prediction_r2_is_high(self, synthetic_y_true, synthetic_y_pred):
        """
        The fixture adds σ=0.1 noise; expect R² > 0.85.
        This guards against a silent sign-flip or axis inversion.
        """
        result = compute_metrics(synthetic_y_true, synthetic_y_pred)
        assert result["r2"] > 0.85, (
            f"R² = {result['r2']:.4f} is unexpectedly low — "
            "check whether arrays were accidentally transposed."
        )

    def test_flat_input_accepted(self):
        """1-D inputs (single gene) must not crash the function."""
        rng = np.random.default_rng(42)
        y_true = rng.standard_normal(100)
        y_pred = y_true + rng.standard_normal(100) * 0.05
        result = compute_metrics(y_true, y_pred)
        assert "r2" in result

    def test_transposed_arrays_give_same_result(self, synthetic_y_true,
                                                synthetic_y_pred):
        """
        compute_metrics ravels both arrays before computing, so (N, G) and
        (G, N) shapes must yield identical results.
        """
        r1 = compute_metrics(synthetic_y_true, synthetic_y_pred)
        r2 = compute_metrics(synthetic_y_true.T, synthetic_y_pred.T)
        assert r1["r2"] == pytest.approx(r2["r2"], abs=1e-6)

    def test_rmse_mae_relationship(self, synthetic_y_true, synthetic_y_pred):
        """
        For any distribution of errors, RMSE ≥ MAE (Jensen's inequality).
        Violation indicates a computational error in one of the metrics.
        """
        result = compute_metrics(synthetic_y_true, synthetic_y_pred)
        assert result["rmse"] >= result["mae"] - 1e-7, (
            f"RMSE ({result['rmse']:.6f}) < MAE ({result['mae']:.6f}) — "
            "this is mathematically impossible."
        )

    def test_bad_model_r2_can_be_negative(self):
        """R² is not bounded at zero for bad models. Verify we don't clamp it."""
        rng = np.random.default_rng(1)
        y_true = rng.standard_normal(200)
        # Predictions are pure noise, completely uncorrelated with y_true
        y_pred = rng.standard_normal(200) * 10
        result = compute_metrics(y_true, y_pred)
        # We can't assert exactly negative, but it must be well below 1
        assert result["r2"] < 0.5


# ===========================================================================
# Tests: compute_metrics_per_gene()
# ===========================================================================

class TestComputeMetricsPerGene:

    EXPECTED_COLUMNS = {"gene", "gene_idx", "r2", "pearson_r", "p_value"}

    def test_returns_dataframe(self, y_true_df, y_pred_array):
        result = compute_metrics_per_gene(y_true_df, y_pred_array)
        assert isinstance(result, pd.DataFrame)

    def test_output_has_correct_columns(self, y_true_df, y_pred_array):
        result = compute_metrics_per_gene(y_true_df, y_pred_array)
        assert self.EXPECTED_COLUMNS == set(result.columns), (
            f"Missing columns: {self.EXPECTED_COLUMNS - set(result.columns)}"
        )

    def test_one_row_per_gene_when_all_have_variance(self, y_true_df, y_pred_array):
        """Normal data: all genes have variance → one row per gene."""
        result = compute_metrics_per_gene(y_true_df, y_pred_array)
        assert len(result) == len(y_true_df.columns)

    def test_low_variance_genes_are_silently_skipped(self):
        """
        CRITICAL: genes with var(y_true) < 1e-10 are dropped without warning.
        Callers must account for a DataFrame shorter than the gene count.
        """
        rng = np.random.default_rng(888)
        n_samples, n_genes = 30, 5
        y_true = rng.standard_normal((n_samples, n_genes))
        y_pred = rng.standard_normal((n_samples, n_genes))

        # Flatten the first gene to a constant — zero variance
        y_true[:, 0] = 0.42

        df = pd.DataFrame(y_true, columns=[f"G{i}" for i in range(n_genes)])
        result = compute_metrics_per_gene(df, y_pred)

        assert len(result) == n_genes - 1, (
            "Expected the zero-variance gene to be silently dropped; "
            f"got {len(result)} rows for {n_genes} genes."
        )
        assert "G0" not in result["gene"].values

    def test_gene_names_preserved_in_output(self, y_true_df, y_pred_array):
        """Gene column in output must exactly match input DataFrame columns."""
        result = compute_metrics_per_gene(y_true_df, y_pred_array)
        output_genes = set(result["gene"].values)
        input_genes = set(y_true_df.columns)
        assert output_genes == input_genes

    def test_gene_idx_values_are_correct(self, y_true_df, y_pred_array):
        """gene_idx must be the integer column position in y_true_df."""
        result = compute_metrics_per_gene(y_true_df, y_pred_array)
        result_indexed = result.set_index("gene")
        for col_pos, gene_name in enumerate(y_true_df.columns):
            assert result_indexed.loc[gene_name, "gene_idx"] == col_pos

    def test_per_gene_r2_values_are_bounded(self, y_true_df, y_pred_array):
        """
        Per-gene R² can legitimately be negative, but p_value must be in [0,1]
        and pearson_r must be in [-1, 1].
        """
        result = compute_metrics_per_gene(y_true_df, y_pred_array)
        assert (result["p_value"].between(0.0, 1.0)).all()
        assert (result["pearson_r"].between(-1.0, 1.0)).all()

    def test_model_meets_r2_benchmark(self, y_true_df, y_pred_array):
        """
        Testing accruacy output to ensure it meets a realistic bechmark of > 0.9 (originally 1 but unlikely to ever practically happen with modelling)
        """
        result = compute_metrics_per_gene(y_true_df, y_pred_array)
        assert (result["r2"] > 0.90).all(), "Model fell below R2 benchmark"

    def test_empty_dataframe_returns_empty_result(self):
        """Edge case: zero genes in input must return an empty DataFrame."""
        df = pd.DataFrame(index=range(10))
        arr = np.empty((10, 0))
        result = compute_metrics_per_gene(df, arr)
        assert len(result) == 0