"""
====================================
Integration tests that validate the real data files load with the correct
shapes, column schemas, and content constraints.

These tests skip cleanly when data_config.json is absent or the data path
does not exist on the current machine. They are intended to be run once on
first setup of a new machine, and whenever the underlying data files are
updated, to catch format regressions before they propagate through the
pipeline.

Weak chains targeted
--------------------
1. network(full).tsv column names: the pipeline assumes exactly TF, Gene,
   Interaction. A renamed column (e.g. "source", "target") silently produces
   an empty feature set. The RNN reconstructor also depends on 'Interaction'
   being numeric, not a string.
2. TF expression and gene expression sample alignment: both TSVs must share
   an identical set of sample IDs in the same order for train_test_split to
   produce correctly paired X, y arrays.
3. External validation cohort shape: the pipeline assumes ~262 samples and
   requires that gene expression columns are a subset of the training
   gene expression columns.
4. Data files must be free of NaN values that would silently corrupt all
   downstream metrics (NaN propagates through pearsonr as NaN, which is
   a plausible-looking float).
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


# extends timeount threshold to 300s/5min to accomodate longer wait to load heavy data files
pytestmark = pytest.mark.timeout(300)

# ---------------------------------------------------------------------------
# Expected data shapes (from README.md and production notebook outputs)
# ---------------------------------------------------------------------------
EXPECTED_SHAPES = {
    "TF(full).tsv":                 {"min_rows": 15000, "min_cols": 1000},
    "Geneexpression (full).tsv":    {"min_rows": 15000, "min_cols": 16000},
    "network(full).tsv":            {"min_rows": 1000000, "min_cols": 3},
    "Liver_bulk_external.tsv":      {"min_rows": 250, "min_cols": 16000},
}

REQUIRED_NETWORK_COLUMNS = {"TF", "Gene", "Interaction"}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def data_files(data_root):
    """Return a dict of {filename: Path} for all expected data files."""
    full_data = data_root / "Full data files"
    if not full_data.exists():
        pytest.skip(
            f"'Full data files' subdirectory not found inside DATA_ROOT ({data_root}). "
            "Ensure the data directory structure matches the README."
        )
    files = {}
    for name in EXPECTED_SHAPES:
        p = full_data / name
        if not p.exists():
            pytest.skip(f"Required data file not found: {p}")
        files[name] = p
    return files


@pytest.fixture(scope="module")
def tf_expression(data_files):
    return pd.read_csv(data_files["TF(full).tsv"], sep="\t", header=0, index_col=0)


@pytest.fixture(scope="module")
def gene_expression(data_files):
    return pd.read_csv(data_files["Geneexpression (full).tsv"], sep="\t",
                       header=0, index_col=0)


@pytest.fixture(scope="module")
def network(data_files):
    return pd.read_csv(data_files["network(full).tsv"], sep="\t")


@pytest.fixture(scope="module")
def external_validation(data_files):
    return pd.read_csv(data_files["Liver_bulk_external.tsv"], sep="\t",
                       header=0, index_col=0)


# ===========================================================================
# Tests: network(full).tsv
# ===========================================================================

class TestNetworkFile:

    def test_required_columns_present(self, network):
        """
        The pipeline reads net['TF'], net['Gene'], net['Interaction'] directly.
        A missing or renamed column produces an empty feature set with no error.
        """
        assert REQUIRED_NETWORK_COLUMNS.issubset(set(network.columns)), (
            f"network(full).tsv is missing required columns. "
            f"Found: {set(network.columns)}. "
            f"Required: {REQUIRED_NETWORK_COLUMNS}"
        )

    def test_minimum_row_count(self, network):
        assert len(network) >= EXPECTED_SHAPES["network(full).tsv"]["min_rows"], (
            f"Network has only {len(network)} rows; expected ≥ 1,000,000."
        )

    def test_interaction_column_is_numeric(self, network):
        """
        format_network() in RNN_reconstructor.py calls pd.to_numeric on
        Interaction. If the column already has the correct dtype, NaN-coercion
        from non-numeric strings will be visible here.
        """
        interaction_numeric = pd.to_numeric(network["Interaction"], errors="coerce")
        n_nan = interaction_numeric.isna().sum()
        assert n_nan == 0, (
            f"Interaction column contains {n_nan} non-numeric values. "
            "These will be silently coerced to NaN by format_network()."
        )

    def test_interaction_values_are_plus_or_minus_one(self, network):
        """
        The RNN edge_MOA array is built from edge_MOA==1 and edge_MOA==-1.
        Any other value (e.g. 0, 2) will be ignored, silently shrinking the network.
        """
        valid_values = {1, -1}
        actual_values = set(network["Interaction"].unique())
        unexpected = actual_values - valid_values
        assert len(unexpected) == 0, (
            f"Unexpected Interaction values: {unexpected}. "
            "Only 1 (activation) and -1 (inhibition) are supported."
        )

    @pytest.mark.skip(reason="LEMBAS-RNN model architecture supports and utilizes self-loops.")
    def test_no_self_loops(self, network):
        """A TF regulating itself would create a trivial loop. Verify absence."""
        self_loops = network[network["TF"] == network["Gene"]]
        assert len(self_loops) == 0, (
            f"Found {len(self_loops)} self-loop edges (TF == Gene). "
            "These may cause numerical instability in the BioNet RNN."
        )

    def test_no_null_values(self, network):
        nulls = network[["TF", "Gene", "Interaction"]].isnull().sum()
        assert nulls.sum() == 0, f"Null values found in network:\n{nulls}"


# ===========================================================================
# Tests: TF expression matrix
# ===========================================================================

class TestTFExpression:

    def test_minimum_shape(self, tf_expression):
        exp = EXPECTED_SHAPES["TF(full).tsv"]
        assert tf_expression.shape[0] >= exp["min_rows"]
        assert tf_expression.shape[1] >= exp["min_cols"]

    def test_no_nan_values(self, tf_expression):
        n_nan = tf_expression.isnull().sum().sum()
        assert n_nan == 0, (
            f"TF expression matrix contains {n_nan} NaN values. "
            "These propagate silently through pearsonr as NaN metrics."
        )

    def test_no_all_zero_columns(self, tf_expression):
        """
        An all-zero TF column has zero variance and will corrupt pearsonr.
        This can occur if a TF was added to the network but never profiled.
        """
        zero_cols = (tf_expression == 0).all(axis=0).sum()
        assert zero_cols == 0, (
            f"{zero_cols} TF columns are all-zero. "
            "These will cause NaN in pearsonr and be silently skipped."
        )

    def test_values_are_numeric_float(self, tf_expression):
        assert tf_expression.dtypes.apply(lambda d: np.issubdtype(d, np.number)).all()


# ===========================================================================
# Tests: Gene expression matrix
# ===========================================================================

class TestGeneExpression:

    def test_minimum_shape(self, gene_expression):
        exp = EXPECTED_SHAPES["Geneexpression (full).tsv"]
        assert gene_expression.shape[0] >= exp["min_rows"]
        assert gene_expression.shape[1] >= exp["min_cols"]

    def test_no_nan_values(self, gene_expression):
        n_nan = gene_expression.isnull().sum().sum()
        assert n_nan == 0, (
            f"Gene expression matrix contains {n_nan} NaN values."
        )

    def test_sample_index_matches_tf_expression(self, tf_expression,
                                                 gene_expression):
        """
        CRITICAL: TF and gene expression must share the same sample IDs in
        the same order. A mismatch means train_test_split silently pairs
        the wrong TF profiles with the wrong gene expression targets.
        """
        assert list(tf_expression.index) == list(gene_expression.index), (
            "Sample index of TF expression and gene expression do not match. "
            "train_test_split will produce misaligned X, y pairs."
        )


# ===========================================================================
# Tests: External validation cohort
# ===========================================================================

class TestExternalValidation:

    def test_minimum_shape(self, external_validation):
        exp = EXPECTED_SHAPES["Liver_bulk_external.tsv"]
        assert external_validation.shape[0] >= exp["min_rows"], (
            f"Validation cohort has {external_validation.shape[0]} samples; "
            f"expected ≥ {exp['min_rows']}."
        )

    def test_sample_count_is_approximately_262(self, external_validation):
        """README and SHAP scripts assume exactly 262 validation samples."""
        n = external_validation.shape[0]
        assert 250 <= n <= 280, (
            f"Validation cohort has {n} samples; expected approximately 262. "
            "SHAP configuration constants (RNN_TEST_SAMPLES=262) may be stale."
        )

    def test_gene_columns_overlap_with_training_genes(self, external_validation,
                                                       gene_expression):
        """
        The y_validation construction zero-fills genes missing from the
        external cohort. If there is NO overlap, all y_validation values
        would be zero and metrics would be meaningless.
        """
        overlap = set(external_validation.columns) & set(gene_expression.columns)
        assert len(overlap) > 0, (
            "External validation cohort shares no gene columns with training "
            "gene expression data. y_validation would be entirely zero-filled."
        )

    def test_no_nan_values(self, external_validation):
        n_nan = external_validation.isnull().sum().sum()
        assert n_nan == 0, (
            f"External validation cohort contains {n_nan} NaN values."
        )


# ===========================================================================
# Tests: Cross-file pipeline constraints
# ===========================================================================

class TestCrossFilePipelineConstraints:

    def test_usable_features_count_is_nonzero(self, tf_expression, network):
        """
        The intersection of TF expression columns with network nodes must
        be non-empty. An empty intersection means no features reach any model.
        """
        network_nodes = set(network["TF"]) | set(network["Gene"])
        usable = [tf for tf in tf_expression.columns if tf in network_nodes]
        assert len(usable) > 0, (
            "No TF expression columns intersect with network nodes. "
            "usable_features is empty — the entire pipeline would produce "
            "zero-feature inputs."
        )

    def test_usable_features_count_matches_expected_range(self, tf_expression,
                                                           network):
        """
        Based on known data: ~1,197 TFs intersect the network.
        A count far outside this range indicates a data/network version mismatch.
        """
        network_nodes = set(network["TF"]) | set(network["Gene"])
        usable = [tf for tf in tf_expression.columns if tf in network_nodes]
        assert 1000 <= len(usable) <= 1500, (
            f"Expected ~1,197 usable TF features, got {len(usable)}. "
            "This may indicate a version mismatch between TF(full).tsv "
            "and network(full).tsv."
        )

    def test_alb_and_afp_exist_in_gene_expression(self, gene_expression):
        """
        ALB and AFP are the two genes targeted by the SHAP analysis.
        Their absence would cause find_gene_index() to return None and
        silently skip the entire SHAP computation.
        """
        for gene in ["ALB", "AFP"]:
            assert gene in gene_expression.columns, (
                f"Gene '{gene}' not found in gene expression columns. "
                f"SHAP analysis for {gene} will be silently skipped."
            )