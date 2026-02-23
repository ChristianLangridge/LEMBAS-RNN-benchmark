"""
===========================
Unit tests for the data preprocessing logic shared across all pipeline scripts.

The same pattern appears in four files:
  - run/data preprocessing/model_boilerplate_remote.py
  - config/predictions/model_train_test_predictions.py
  - config/predictions/model_validation_predictions.py
  - config/SHAP/SHAP_generation_baseline.py  (+ RNN variant)

Because it is duplicated rather than imported, a bug fix in one place does
not propagate. These tests verify the *logic* as a contract, so any future
refactor can be validated against them.

Weak chains targeted
--------------------
1. TF filtering: intersection of tf_expression.columns with network_nodes.
   If the network has been updated but TF expression hasn't (or vice versa),
   usable_features silently shrinks to empty.
2. Zero-fill for missing TFs: x_validation is built by filling 0 for any TF
   in usable_features that is absent from the external cohort. Column ordering
   must be exactly usable_features — a sort or shuffle breaks model input.
3. Train/test split determinism: random_state=888 is a cross-script contract.
   Any script that re-splits must produce identical indices.
4. y_train/y_test are numpy arrays; x_train/x_test are also numpy arrays.
   Downstream model.predict() calls depend on this (DataFrames with named
   columns cause silent issues in some sklearn estimators).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Helpers: re-implement the pipeline logic locally for hermetic testing
# ---------------------------------------------------------------------------

def build_usable_features(tf_expression_df: pd.DataFrame,
                          network_df: pd.DataFrame) -> list:
    """
    Replicate the usable_features logic from model_boilerplate_remote.py.
    Returns the list of TF column names that appear in the network.
    """
    network_tfs = set(network_df["TF"].unique())
    network_genes = set(network_df["Gene"].unique())
    network_nodes = network_tfs | network_genes
    return [tf for tf in tf_expression_df.columns if tf in network_nodes]


def build_x_validation(validation_df: pd.DataFrame,
                       usable_features: list) -> pd.DataFrame:
    """
    Replicate the zero-fill x_validation construction from
    model_validation_predictions.py and SHAP_generation_baseline.py.
    """
    present = [f for f in usable_features if f in validation_df.columns]
    missing = [f for f in usable_features if f not in validation_df.columns]

    x_val = pd.DataFrame(index=validation_df.index)
    for feat in present:
        x_val[feat] = validation_df[feat]
    for feat in missing:
        x_val[feat] = 0

    return x_val[usable_features]  # enforce exact column order


# ===========================================================================
# Tests: TF feature filtering (build_usable_features)
# ===========================================================================

class TestUsableFeatures:

    def test_returns_only_tfs_in_network(self, tf_expression_df,
                                         synthetic_network):
        """Only TFs that appear as nodes in the network should be kept."""
        usable = build_usable_features(tf_expression_df, synthetic_network)
        network_nodes = (set(synthetic_network["TF"]) |
                         set(synthetic_network["Gene"]))
        for tf in usable:
            assert tf in network_nodes, (
                f"'{tf}' is in usable_features but not in the network."
            )

    def test_excludes_tfs_not_in_network(self, synthetic_network):
        """TFs not in the network must be excluded, even if in expression data."""
        # Add a TF that is not in the network
        n_samples = 10
        tf_expr = pd.DataFrame(
            np.ones((n_samples, 3)),
            columns=["TF00", "TF01", "ORPHAN_TF"]
        )
        usable = build_usable_features(tf_expr, synthetic_network)
        assert "ORPHAN_TF" not in usable

    def test_empty_result_when_no_overlap(self, synthetic_network):
        """
        CRITICAL: If TF expression columns share no names with the network,
        usable_features is empty. Callers must guard against this.
        """
        tf_expr = pd.DataFrame(
            np.ones((10, 3)),
            columns=["GENE_X", "GENE_Y", "GENE_Z"]
        )
        usable = build_usable_features(tf_expr, synthetic_network)
        assert usable == [], (
            "Expected empty list when TF expression and network share no nodes."
        )

    def test_preserves_original_column_order(self, tf_expression_df,
                                              synthetic_network):
        """
        usable_features must follow tf_expression_df.columns order, not network order.
        Model weights are positionally bound to the feature order seen at training.
        """
        usable = build_usable_features(tf_expression_df, synthetic_network)
        tf_cols_in_network = [
            c for c in tf_expression_df.columns
            if c in (set(synthetic_network["TF"]) | set(synthetic_network["Gene"]))
        ]
        assert usable == tf_cols_in_network

    def test_result_is_list_not_set(self, tf_expression_df, synthetic_network):
        """Return type must be list (ordered), not set (unordered)."""
        usable = build_usable_features(tf_expression_df, synthetic_network)
        assert isinstance(usable, list)

    def test_no_duplicate_features(self, tf_expression_df, synthetic_network):
        """Duplicates in usable_features would corrupt the model input."""
        usable = build_usable_features(tf_expression_df, synthetic_network)
        assert len(usable) == len(set(usable))


# ===========================================================================
# Tests: x_validation zero-fill (build_x_validation)
# ===========================================================================

class TestXValidationZeroFill:

    def test_output_columns_match_usable_features_exactly(
            self, tf_expression_df, synthetic_network):
        """
        The column order in x_validation must exactly match usable_features.
        This is the positional contract with the trained model.
        """
        usable = build_usable_features(tf_expression_df, synthetic_network)
        # Simulate an external cohort with some TFs missing
        partial_cohort = tf_expression_df.iloc[:5].copy()
        partial_cohort = partial_cohort.drop(columns=usable[:2])  # drop first 2

        x_val = build_x_validation(partial_cohort, usable)
        assert list(x_val.columns) == usable

    def test_missing_tfs_are_zero_filled(self, tf_expression_df,
                                          synthetic_network):
        """TFs absent from the external cohort must be filled with exactly 0."""
        usable = build_usable_features(tf_expression_df, synthetic_network)
        partial_cohort = tf_expression_df.iloc[:5].copy()
        dropped = usable[:2]
        partial_cohort = partial_cohort.drop(columns=dropped)

        x_val = build_x_validation(partial_cohort, usable)
        for tf in dropped:
            assert (x_val[tf] == 0).all(), (
                f"Missing TF '{tf}' was not zero-filled — "
                "this will silently corrupt model input."
            )

    def test_present_tfs_retain_original_values(self, tf_expression_df,
                                                 synthetic_network):
        """TFs that ARE present in the cohort must keep their original values."""
        usable = build_usable_features(tf_expression_df, synthetic_network)
        cohort = tf_expression_df.iloc[:5].copy()
        x_val = build_x_validation(cohort, usable)

        for tf in usable:
            if tf in cohort.columns:
                np.testing.assert_array_almost_equal(
                    x_val[tf].values, cohort[tf].values,
                    err_msg=f"Values for present TF '{tf}' were altered."
                )

    def test_output_shape_is_always_n_samples_by_n_features(
            self, tf_expression_df, synthetic_network):
        """Shape must be (n_samples, len(usable_features)) regardless of overlap."""
        usable = build_usable_features(tf_expression_df, synthetic_network)
        n_samples = 8
        cohort = tf_expression_df.iloc[:n_samples].drop(columns=usable[:3])
        x_val = build_x_validation(cohort, usable)
        assert x_val.shape == (n_samples, len(usable))

    def test_all_missing_tfs_gives_zero_matrix(self, synthetic_network):
        """Edge case: external cohort has no TF columns at all → all zeros."""
        usable = list(synthetic_network["TF"].unique())[:4]
        empty_cohort = pd.DataFrame(
            np.ones((5, 2)),
            columns=["UNRELATED_COL_A", "UNRELATED_COL_B"]
        )
        x_val = build_x_validation(empty_cohort, usable)
        assert (x_val.values == 0).all()


# ===========================================================================
# Tests: train/test split determinism (random_state=888)
# ===========================================================================

class TestSplitDeterminism:
    """
    The 80/20 split with random_state=888 is a cross-script contract:
    model_boilerplate_remote.py, model_train_test_predictions.py, and any
    evaluation notebook that recomputes the split must produce identical
    train/test indices. Changing the seed or the split fraction silently
    invalidates all saved predictions.
    """

    SPLIT_SEED = 888
    SPLIT_SIZE = 0.2

    def test_same_seed_gives_same_split(self, tf_expression_df,
                                        gene_expression_df):
        """Two calls with the same seed must produce byte-identical splits."""
        x1_train, x1_test, y1_train, y1_test = train_test_split(
            tf_expression_df, gene_expression_df,
            test_size=self.SPLIT_SIZE, random_state=self.SPLIT_SEED
        )
        x2_train, x2_test, y2_train, y2_test = train_test_split(
            tf_expression_df, gene_expression_df,
            test_size=self.SPLIT_SIZE, random_state=self.SPLIT_SEED
        )
        np.testing.assert_array_equal(x1_train.index, x2_train.index)
        np.testing.assert_array_equal(x1_test.index, x2_test.index)

    def test_different_seed_gives_different_split(self, tf_expression_df,
                                                   gene_expression_df):
        """Sanity check: different seeds must produce at least one different index."""
        _, x_test_888, _, _ = train_test_split(
            tf_expression_df, gene_expression_df,
            test_size=self.SPLIT_SIZE, random_state=888
        )
        _, x_test_42, _, _ = train_test_split(
            tf_expression_df, gene_expression_df,
            test_size=self.SPLIT_SIZE, random_state=42
        )
        assert not np.array_equal(x_test_888.index.values,
                                  x_test_42.index.values), (
            "Seeds 888 and 42 produced the same split — "
            "this strongly suggests the split is not actually seeded."
        )

    def test_split_proportions_are_correct(self, tf_expression_df,
                                           gene_expression_df):
        """Test set must be ~20% of total samples (allowing ±1 for rounding)."""
        total = len(tf_expression_df)
        x_train, x_test, _, _ = train_test_split(
            tf_expression_df, gene_expression_df,
            test_size=self.SPLIT_SIZE, random_state=self.SPLIT_SEED
        )
        expected_test = round(total * self.SPLIT_SIZE)
        assert abs(len(x_test) - expected_test) <= 1
        assert len(x_train) + len(x_test) == total

    def test_x_and_y_indices_are_aligned_after_split(self, tf_expression_df,
                                                       gene_expression_df):
        """
        After splitting, x_train.index must equal y_train.index row-for-row.
        Misalignment means features and targets are paired incorrectly.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            tf_expression_df, gene_expression_df,
            test_size=self.SPLIT_SIZE, random_state=self.SPLIT_SEED
        )
        np.testing.assert_array_equal(x_train.index, y_train.index)
        np.testing.assert_array_equal(x_test.index, y_test.index)

    def test_to_numpy_conversion_preserves_values(self, tf_expression_df,
                                                   gene_expression_df):
        """
        The boilerplate calls .to_numpy() on train/test splits.
        Verify values are identical before and after conversion.
        """
        x_train_df, _, y_train_df, _ = train_test_split(
            tf_expression_df, gene_expression_df,
            test_size=self.SPLIT_SIZE, random_state=self.SPLIT_SEED
        )
        x_train_np = x_train_df.to_numpy()
        y_train_np = y_train_df.to_numpy()

        np.testing.assert_array_almost_equal(x_train_df.values, x_train_np)
        np.testing.assert_array_almost_equal(y_train_df.values, y_train_np)

    def test_no_sample_overlap_between_train_and_test(self, tf_expression_df,
                                                       gene_expression_df):
        """Absolute requirement: no sample may appear in both train and test."""
        x_train, x_test, _, _ = train_test_split(
            tf_expression_df, gene_expression_df,
            test_size=self.SPLIT_SIZE, random_state=self.SPLIT_SEED
        )
        overlap = set(x_train.index) & set(x_test.index)
        assert len(overlap) == 0, (
            f"Data leakage: {len(overlap)} samples appear in both "
            "train and test sets."
        )