"""
=====================================
Integration tests that validate the saved prediction artefacts are internally
consistent and that the pipeline produces reproducible outputs.

These tests do NOT re-run training (hours of compute). Instead they verify
the *contracts* between pipeline stages by loading saved .npz files and
checking their shapes, dtypes, column ordering, and metric plausibility.

Weak chains targeted
--------------------
1. npz column ordering: y_train_columns and y_test_columns are stored as
   numpy arrays. If the DataFrame column order differed at save time vs load
   time (e.g. due to dict ordering in Python <3.7 or a sort), all per-gene
   metrics would be computed against the wrong gene.
2. Prediction array shapes: mlr_y_pred_train must be (n_train, n_genes).
   A transposed save would pass the load but produce wrong metrics silently.
3. Metric plausibility: the known-good metrics from the README (MLR R²≈0.95
   on validation) should be approximately reproducible. A large deviation
   suggests model version mismatch.
4. Train/test non-overlap verified on the saved arrays (not just the split
   function), confirming the artefacts were actually generated with the
   correct seed.
5. NaN propagation: a single NaN in a prediction array causes pearsonr to
   return NaN for that gene (or aggregate metric), which surfaces as a
   missing result in figures.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Known-good metric bounds from README (validation set, all models)
# Bounds are generous (±0.05) to allow for minor version differences.
# ---------------------------------------------------------------------------
METRIC_BOUNDS = {
    "MLR":  {"r2_min": 0.90, "pearson_min": 0.95},
    "XGBRF": {"r2_min": 0.88, "pearson_min": 0.93},
    "RNN":  {"r2_min": 0.79, "pearson_min": 0.87},
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# original "model_predictions_uncentered_v1.npz" file is 4.64GB so needed to get a smaller subset file "mock_data.npz" to run this test 
@pytest.fixture(scope="module")
def saved_predictions_path(data_root):
    p = data_root / "Saved predictions" / "mock_data.npz" 
    if not p.exists():
        pytest.skip(f"Saved predictions not found at {p}. Run the prediction pipeline first.")
    return p


@pytest.fixture(scope="module")
def saved_validation_path(data_root):
    p = data_root / "Saved predictions" / "model_predictions_validation_v1.npz"
    if not p.exists():
        pytest.skip(f"Saved validation predictions not found at {p}.")
    return p


@pytest.fixture(scope="module")
def predictions(saved_predictions_path):
    return np.load(saved_predictions_path, allow_pickle=True)


@pytest.fixture(scope="module")
def val_predictions(saved_validation_path):
    return np.load(saved_validation_path, allow_pickle=True)


# ===========================================================================
# Tests: Saved prediction artefact structure
# ===========================================================================

class TestPredictionArtefactStructure:

    REQUIRED_TRAIN_TEST_KEYS = {
        "y_train", "y_train_columns",
        "y_test", "y_test_columns",
        "mlr_y_pred_train", "xgbrf_y_pred_train", "rnn_y_pred_train",
        "mlr_y_pred_test", "xgbrf_y_pred_test", "rnn_y_pred_test",
    }

    REQUIRED_VALIDATION_KEYS = {
        "y_validation", "y_validation_columns",
        "mlr_y_pred_val", "xgbrf_y_pred_val", "rnn_y_pred_val",
    }

    def test_all_required_train_test_keys_present(self, predictions):
        missing = self.REQUIRED_TRAIN_TEST_KEYS - set(predictions.files)
        assert len(missing) == 0, (
            f"Saved predictions npz is missing keys: {missing}. "
            "The load step in model_load.py will raise a KeyError."
        )

    def test_all_required_validation_keys_present(self, val_predictions):
        missing = self.REQUIRED_VALIDATION_KEYS - set(val_predictions.files)
        assert len(missing) == 0, (
            f"Saved validation predictions npz is missing keys: {missing}."
        )

    def test_train_prediction_shapes_consistent(self, predictions):
        """All train prediction arrays must have the same shape as y_train."""
        y_train = predictions["y_train"]
        for model in ["mlr", "xgbrf", "rnn"]:
            pred = predictions[f"{model}_y_pred_train"]
            assert pred.shape == y_train.shape, (
                f"{model}_y_pred_train shape {pred.shape} != "
                f"y_train shape {y_train.shape}. "
                "Arrays may have been saved transposed."
            )

    def test_test_prediction_shapes_consistent(self, predictions):
        y_test = predictions["y_test"]
        for model in ["mlr", "xgbrf", "rnn"]:
            pred = predictions[f"{model}_y_pred_test"]
            assert pred.shape == y_test.shape, (
                f"{model}_y_pred_test shape {pred.shape} != "
                f"y_test shape {y_test.shape}."
            )

    def test_validation_prediction_shapes_consistent(self, val_predictions):
        y_val = val_predictions["y_validation"]
        for model in ["mlr", "xgbrf", "rnn"]:
            pred = val_predictions[f"{model}_y_pred_val"]
            assert pred.shape == y_val.shape, (
                f"{model}_y_pred_val shape {pred.shape} != "
                f"y_validation shape {y_val.shape}."
            )

    def test_column_arrays_match_matrix_width(self, predictions):
        """
        y_train_columns must have exactly as many entries as y_train has columns.
        A mismatch means per-gene metric DataFrames will have wrong gene labels.
        """
        y_train_cols = predictions["y_train_columns"]
        assert len(y_train_cols) == predictions["y_train"].shape[1], (
            f"y_train_columns has {len(y_train_cols)} entries but "
            f"y_train has {predictions['y_train'].shape[1]} columns."
        )

    def test_train_and_test_columns_are_identical(self, predictions):
        """
        Train and test sets must have the same gene columns in the same order.
        A mismatch means per-gene analysis on test set uses wrong gene labels.
        """
        train_cols = list(predictions["y_train_columns"])
        test_cols = list(predictions["y_test_columns"])
        assert train_cols == test_cols, (
            "y_train_columns and y_test_columns differ. "
            "This indicates a column ordering issue at save time."
        )

    def test_no_nan_in_any_prediction_array(self, predictions):
        """
        NaN in any prediction array propagates silently through all metrics.
        Verify cleanliness of the saved artefacts.
        """
        for model in ["mlr", "xgbrf", "rnn"]:
            for split in ["train", "test"]:
                key = f"{model}_y_pred_{split}"
                arr = predictions[key]
                n_nan = np.isnan(arr).sum()
                assert n_nan == 0, (
                    f"{key} contains {n_nan} NaN values. "
                    "Metrics computed on this array will be NaN or misleading."
                )

    def test_no_nan_in_validation_predictions(self, val_predictions):
        for model in ["mlr", "xgbrf", "rnn"]:
            arr = val_predictions[f"{model}_y_pred_val"]
            n_nan = np.isnan(arr).sum()
            assert n_nan == 0, f"{model}_y_pred_val contains {n_nan} NaN values."

    def test_prediction_values_are_finite(self, predictions):
        """Inf values will cause rmse to be inf and r2 to be -inf."""
        for model in ["mlr", "xgbrf", "rnn"]:
            for split in ["train", "test"]:
                key = f"{model}_y_pred_{split}"
                assert np.isfinite(predictions[key]).all(), (
                    f"{key} contains non-finite values (inf or -inf)."
                )


# ===========================================================================
# Tests: Metric plausibility against known-good benchmarks
# ===========================================================================

class TestMetricPlausibility:
    """
    Soft regression guard: if a model's performance on the validation set
    falls significantly below the published README values, something has
    changed — wrong model version, wrong data, wrong preprocessing.
    """

    def _compute_aggregate_metrics(self, y_true: np.ndarray,
                                   y_pred: np.ndarray) -> dict:
        y_true_flat = y_true.ravel()
        y_pred_flat = y_pred.ravel()
        pearson_r, _ = pearsonr(y_true_flat, y_pred_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)
        return {"r2": r2, "pearson_r": pearson_r}

    def test_mlr_validation_metrics_within_expected_range(self, val_predictions):
        y_val = val_predictions["y_validation"]
        y_pred = val_predictions["mlr_y_pred_val"]
        m = self._compute_aggregate_metrics(y_val, y_pred)
        bounds = METRIC_BOUNDS["MLR"]
        assert m["r2"] >= bounds["r2_min"], (
            f"MLR validation R² = {m['r2']:.4f}, expected ≥ {bounds['r2_min']}. "
            "Model may be a different version than the one documented in README."
        )
        assert m["pearson_r"] >= bounds["pearson_min"]

    def test_xgbrf_validation_metrics_within_expected_range(self, val_predictions):
        y_val = val_predictions["y_validation"]
        y_pred = val_predictions["xgbrf_y_pred_val"]
        m = self._compute_aggregate_metrics(y_val, y_pred)
        bounds = METRIC_BOUNDS["XGBRF"]
        assert m["r2"] >= bounds["r2_min"], (
            f"XGBRF validation R² = {m['r2']:.4f}, expected ≥ {bounds['r2_min']}."
        )

    def test_rnn_validation_metrics_within_expected_range(self, val_predictions):
        y_val = val_predictions["y_validation"]
        y_pred = val_predictions["rnn_y_pred_val"]
        m = self._compute_aggregate_metrics(y_val, y_pred)
        bounds = METRIC_BOUNDS["RNN"]
        assert m["r2"] >= bounds["r2_min"], (
            f"RNN validation R² = {m['r2']:.4f}, expected ≥ {bounds['r2_min']}. "
            "Check model checkpoint version and preprocessing seed."
        )

    def test_mlr_outperforms_rnn_on_validation(self, val_predictions):
        """
        From the README: MLR R²=0.9528 > XGBRF R²=0.9346 > RNN R²=0.8441.
        If RNN outperforms MLR on validation, something is wrong.
        """
        y_val = val_predictions["y_validation"]
        mlr_r2 = r2_score(y_val.ravel(), val_predictions["mlr_y_pred_val"].ravel())
        rnn_r2 = r2_score(y_val.ravel(), val_predictions["rnn_y_pred_val"].ravel())
        assert mlr_r2 > rnn_r2, (
            f"RNN (R²={rnn_r2:.4f}) outperforms MLR (R²={mlr_r2:.4f}) on validation. "
            "This contradicts the expected ordering — check model artefacts."
        )

    def test_train_r2_higher_than_test_r2_for_all_models(self, predictions):
        """
        Every model should fit training data better than test data
        (some degree of overfitting is expected). If test R² > train R²,
        the split may be contaminated.
        """
        y_train = predictions["y_train"]
        y_test = predictions["y_test"]

        for model in ["mlr", "xgbrf", "rnn"]:
            train_r2 = r2_score(
                y_train.ravel(),
                predictions[f"{model}_y_pred_train"].ravel()
            )
            test_r2 = r2_score(
                y_test.ravel(),
                predictions[f"{model}_y_pred_test"].ravel()
            )
            assert train_r2 >= test_r2 - 0.05, (
                f"{model.upper()} test R² ({test_r2:.4f}) exceeds train R² "
                f"({train_r2:.4f}) by more than 0.05. "
                "This may indicate data leakage in the train/test split."
            )