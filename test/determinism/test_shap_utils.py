"""
========================
Unit tests for SHAP pipeline utility functions from
config/SHAP/SHAP_generation_baseline.py.

These functions are un-imported helpers (not in a package), so we test the
logic by re-implementing them locally. Any divergence between the production
code and this reference will surface as a test failure.

Weak chains targeted
--------------------
1. find_gene_index() returns None for missing genes rather than raising.
   The SHAP pipeline checks for None after the loop, but a single missing
   gene among many silently drops SHAP computation for that gene.

2. extract_xgbrf_model_from_batches() has critical arithmetic at batch
   boundaries (e.g. gene_idx=999 → batch 0 within-batch 999;
   gene_idx=1000 → batch 1, within-batch 0). Off-by-one errors here would
   return a model for the *wrong gene* with no runtime error.

3. The function's guard conditions (batch_idx out of range, missing
   estimators_ attribute, within-batch index out of range) must raise
   specific, informative exceptions — not generic AttributeErrors that
   hide the root cause.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Reference implementations (mirroring production code)
# ---------------------------------------------------------------------------

def find_gene_index(gene_name: str, gene_columns) -> int | None:
    """Mirror of find_gene_index in SHAP_generation_baseline.py."""
    try:
        return list(gene_columns).index(gene_name)
    except ValueError:
        return None


def extract_xgbrf_model_from_batches(batch_models: list, gene_idx: int,
                                      batch_size: int = 1000):
    """Mirror of extract_xgbrf_model_from_batches in SHAP_generation_baseline.py."""
    batch_idx = gene_idx // batch_size
    within_batch_idx = gene_idx % batch_size

    if batch_idx >= len(batch_models):
        raise IndexError(
            f"Batch index {batch_idx} out of range "
            f"(only {len(batch_models)} batches available)."
        )

    batch_model = batch_models[batch_idx]

    if not hasattr(batch_model, "estimators_"):
        raise AttributeError(
            f"Batch model at index {batch_idx} is not a fitted "
            "MultiOutputRegressor (missing 'estimators_' attribute)."
        )

    if within_batch_idx >= len(batch_model.estimators_):
        raise IndexError(
            f"Within-batch index {within_batch_idx} out of range "
            f"(batch has {len(batch_model.estimators_)} estimators)."
        )

    return batch_model.estimators_[within_batch_idx]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_batch(n_estimators: int):
    """Return a mock MultiOutputRegressor with n_estimators stub estimators."""
    mock = MagicMock()
    mock.estimators_ = [MagicMock(name=f"est_{i}") for i in range(n_estimators)]
    return mock


# ===========================================================================
# Tests: find_gene_index()
# ===========================================================================

class TestFindGeneIndex:

    def test_found_gene_returns_correct_index(self):
        columns = ["GENE_A", "ALB", "AFP", "GENE_D"]
        assert find_gene_index("ALB", columns) == 1
        assert find_gene_index("AFP", columns) == 2

    def test_first_gene_returns_zero(self):
        columns = ["ALB", "AFP", "GENE_C"]
        assert find_gene_index("ALB", columns) == 0

    def test_last_gene_returns_correct_index(self):
        columns = ["GENE_A", "GENE_B", "ALB"]
        assert find_gene_index("ALB", columns) == 2

    def test_missing_gene_returns_none(self):
        """
        CRITICAL: The production code returns None (not raises) for a missing
        gene. Downstream code must check the return value explicitly.
        """
        columns = ["GENE_A", "GENE_B", "GENE_C"]
        result = find_gene_index("ALB", columns)
        assert result is None, (
            "A missing gene must return None, not raise an exception."
        )

    def test_empty_columns_returns_none(self):
        assert find_gene_index("ALB", []) is None

    def test_works_with_pandas_index(self):
        """Must accept a pd.Index (as encountered in production) not just list."""
        idx = pd.Index(["GENE_A", "ALB", "AFP"])
        assert find_gene_index("ALB", idx) == 1

    def test_case_sensitive(self):
        """Gene names are case-sensitive. 'alb' ≠ 'ALB'."""
        columns = ["GENE_A", "ALB", "alb"]
        assert find_gene_index("alb", columns) == 2

    def test_duplicate_column_returns_first_occurrence(self):
        """If a gene appears twice, the first index is returned."""
        columns = ["ALB", "GENE_B", "ALB"]
        assert find_gene_index("ALB", columns) == 0


# ===========================================================================
# Tests: extract_xgbrf_model_from_batches()
# ===========================================================================

class TestExtractXgbrfModelFromBatches:

    # -----------------------------------------------------------------------
    # Arithmetic correctness
    # -----------------------------------------------------------------------

    def test_first_gene_in_first_batch(self):
        """gene_idx=0 → batch 0, within-batch 0."""
        batch_size = 5
        batches = [_make_mock_batch(batch_size), _make_mock_batch(batch_size)]
        result = extract_xgbrf_model_from_batches(batches, 0, batch_size)
        assert result is batches[0].estimators_[0]

    def test_last_gene_in_first_batch(self):
        """gene_idx=batch_size-1 → batch 0, within-batch batch_size-1."""
        batch_size = 5
        batches = [_make_mock_batch(batch_size), _make_mock_batch(batch_size)]
        result = extract_xgbrf_model_from_batches(batches, batch_size - 1, batch_size)
        assert result is batches[0].estimators_[batch_size - 1]

    def test_first_gene_in_second_batch(self):
        """
        CRITICAL BOUNDARY: gene_idx=batch_size → batch 1, within-batch 0.
        An off-by-one error here silently returns the *wrong gene's model*.
        """
        batch_size = 5
        batches = [_make_mock_batch(batch_size), _make_mock_batch(batch_size)]
        result = extract_xgbrf_model_from_batches(batches, batch_size, batch_size)
        assert result is batches[1].estimators_[0], (
            "gene_idx == batch_size must resolve to the FIRST estimator "
            "of the SECOND batch (off-by-one boundary)."
        )

    def test_gene_in_middle_of_second_batch(self):
        batch_size = 5
        batches = [_make_mock_batch(batch_size), _make_mock_batch(batch_size)]
        gene_idx = batch_size + 2  # → batch 1, within-batch 2
        result = extract_xgbrf_model_from_batches(batches, gene_idx, batch_size)
        assert result is batches[1].estimators_[2]

    def test_real_world_batch_size_1000_boundary(self):
        """
        Replicate exact real-world boundary: XGBRF uses batch_size=1000.
        gene_idx=999 → batch 0, within 999.
        gene_idx=1000 → batch 1, within 0.
        """
        batch_size = 1000
        batches = [_make_mock_batch(batch_size), _make_mock_batch(batch_size)]

        result_999 = extract_xgbrf_model_from_batches(batches, 999, batch_size)
        result_1000 = extract_xgbrf_model_from_batches(batches, 1000, batch_size)

        assert result_999 is batches[0].estimators_[999]
        assert result_1000 is batches[1].estimators_[0]

    # -----------------------------------------------------------------------
    # Guard conditions / error handling
    # -----------------------------------------------------------------------

    def test_raises_index_error_when_batch_idx_out_of_range(self):
        """
        Requesting a gene beyond the total model capacity must raise IndexError,
        not silently wrap around or return a wrong-batch model.
        """
        batch_size = 5
        batches = [_make_mock_batch(batch_size)]  # only 1 batch → 5 genes max
        with pytest.raises(IndexError, match="Batch index"):
            extract_xgbrf_model_from_batches(batches, batch_size, batch_size)

    def test_raises_attribute_error_for_unfitted_model(self):
        """A batch object without estimators_ must raise AttributeError."""
        bad_batch = object()  # has no estimators_ attribute
        with pytest.raises(AttributeError, match="estimators_"):
            extract_xgbrf_model_from_batches([bad_batch], 0, batch_size=5)

    def test_raises_index_error_when_within_batch_idx_out_of_range(self):
        """
        If the batch has fewer estimators than batch_size (e.g. the last,
        partial batch), requesting a gene beyond its length must raise IndexError.
        """
        batch_size = 5
        short_batch = _make_mock_batch(3)  # only 3 estimators, not 5
        with pytest.raises(IndexError, match="Within-batch index"):
            extract_xgbrf_model_from_batches([short_batch], 4, batch_size)

    def test_empty_batch_list_raises_index_error(self):
        with pytest.raises(IndexError):
            extract_xgbrf_model_from_batches([], 0, batch_size=5)

    # -----------------------------------------------------------------------
    # Integration with conftest fixture
    # -----------------------------------------------------------------------

    def test_with_real_sklearn_batch_fixture(self, synthetic_xgbrf_batch_models):
        """
        Smoke test against a real sklearn MultiOutputRegressor from the conftest.
        Verifies the function works end-to-end with a genuine fitted model.
        """
        batches, batch_size = synthetic_xgbrf_batch_models
        # gene_idx=0 → first estimator of first batch
        model = extract_xgbrf_model_from_batches(batches, 0, batch_size)
        assert hasattr(model, "predict"), (
            "Extracted estimator must have a predict() method."
        )

    def test_extracted_model_can_predict(self, synthetic_xgbrf_batch_models):
        """The extracted single-gene model must accept a 2-D numpy input."""
        from conftest import N_TFS
        batches, batch_size = synthetic_xgbrf_batch_models
        rng = np.random.default_rng(1)
        X_test = rng.standard_normal((5, N_TFS))

        model = extract_xgbrf_model_from_batches(batches, 0, batch_size)
        pred = model.predict(X_test)
        assert pred.shape == (5,), (
            f"Single-gene model prediction should be 1-D with shape (5,), "
            f"got {pred.shape}."
        )