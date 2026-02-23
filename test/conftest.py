"""
conftest.py
===========
Shared pytest fixtures and configuration for the LEMBAS-RNN benchmark test suite.

Design principles
-----------------
- Unit tests have zero external dependencies: all fixtures are synthetic,
  dimensionally faithful to the real data, and deterministic (seed=888).
- Integration tests require a valid data_config.json and real data files.
  They are marked @pytest.mark.integration and skipped cleanly when data
  is absent — never fail with a cryptic FileNotFoundError.
- Fixtures are scoped as narrowly as possible (function scope by default)
  so tests never share mutable state.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Repo root resolution
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    """Walk upward from this file until README.md is found."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "README.md").exists():
            return parent
    raise RuntimeError(
        "Could not locate repo root (no README.md found in any parent directory). "
        "Run pytest from inside the repository."
    )


REPO_ROOT = _find_repo_root()

# Ensure repo modules are importable in all test processes
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Custom markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests that require real data files (skipped if data absent)",
    )


# ---------------------------------------------------------------------------
# Integration test: DATA_ROOT resolution
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def data_root():
    """
    Return DATA_ROOT from data_config.json.

    Skips the test (not fails) if:
      - data_config.json does not exist
      - DATA_ROOT key is missing
      - The resolved path does not exist on disk
    """
    config_path = REPO_ROOT / "data_config.json"

    if not config_path.exists():
        pytest.skip(
            "data_config.json not found at repo root. "
            "Integration tests require a configured data path."
        )

    with open(config_path) as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as exc:
            pytest.skip(f"data_config.json is malformed JSON: {exc}")

    root = cfg.get("DATA_ROOT")
    if not root:
        pytest.skip("data_config.json is missing the 'DATA_ROOT' key.")

    root_path = Path(root)
    if not root_path.exists():
        pytest.skip(
            f"DATA_ROOT '{root}' does not exist on this machine. "
            "Update data_config.json to point to your local data directory."
        )

    return root_path


# ---------------------------------------------------------------------------
# Synthetic data fixtures (unit tests — no disk I/O)
# ---------------------------------------------------------------------------

# Dimensions chosen to be small but representative of the real structure:
#   real data: ~15,935 samples × 1,197 TFs × 16,100 genes
#   synthetic: 50 samples  × 10 TFs    × 20 genes
N_SAMPLES  = 50
N_TFS      = 10
N_GENES    = 20
RNG        = np.random.default_rng(888)

TF_NAMES   = [f"TF{i:02d}" for i in range(N_TFS)]
GENE_NAMES = [f"GENE{i:03d}" for i in range(N_GENES)]


@pytest.fixture()
def synthetic_y_true() -> np.ndarray:
    """Ground-truth expression matrix: (N_SAMPLES, N_GENES)."""
    return RNG.standard_normal((N_SAMPLES, N_GENES)).astype(np.float32)


@pytest.fixture()
def synthetic_y_pred(synthetic_y_true) -> np.ndarray:
    """
    Prediction that is strongly correlated with ground truth
    (adds small Gaussian noise), giving R² ≈ 0.95.
    """
    noise = RNG.standard_normal(synthetic_y_true.shape).astype(np.float32) * 0.1
    return synthetic_y_true + noise


@pytest.fixture()
def perfect_y_pred(synthetic_y_true) -> np.ndarray:
    """Exact copy of ground truth — used to verify perfect-score edge cases."""
    return synthetic_y_true.copy()


@pytest.fixture()
def y_true_df(synthetic_y_true) -> pd.DataFrame:
    """Ground-truth as a labelled DataFrame (matches compute_metrics_per_gene API)."""
    return pd.DataFrame(synthetic_y_true, columns=GENE_NAMES)


@pytest.fixture()
def y_pred_array(synthetic_y_pred) -> np.ndarray:
    """Predictions as a plain numpy array (matches compute_metrics_per_gene API)."""
    return synthetic_y_pred


@pytest.fixture()
def synthetic_network() -> pd.DataFrame:
    """
    Minimal network TSV in the format expected by the pipeline.

    Columns: TF, Gene, Interaction
    Covers the first 5 TFs and first 10 genes with alternating activation/inhibition.
    """
    rows = []
    for i, tf in enumerate(TF_NAMES[:5]):
        for j, gene in enumerate(GENE_NAMES[:10]):
            interaction = 1 if (i + j) % 2 == 0 else -1
            rows.append({"TF": tf, "Gene": gene, "Interaction": interaction})
    return pd.DataFrame(rows)


@pytest.fixture()
def tf_expression_df() -> pd.DataFrame:
    """TF expression matrix: (N_SAMPLES, N_TFS), indexed by sample IDs."""
    data = RNG.standard_normal((N_SAMPLES, N_TFS)).astype(np.float32)
    index = [f"sample_{i}" for i in range(N_SAMPLES)]
    return pd.DataFrame(data, index=index, columns=TF_NAMES)


@pytest.fixture()
def gene_expression_df() -> pd.DataFrame:
    """Gene expression matrix: (N_SAMPLES, N_GENES), indexed by sample IDs."""
    data = RNG.standard_normal((N_SAMPLES, N_GENES)).astype(np.float32)
    index = [f"sample_{i}" for i in range(N_SAMPLES)]
    return pd.DataFrame(data, index=index, columns=GENE_NAMES)


@pytest.fixture()
def synthetic_xgbrf_batch_models():
    """
    Mock of the XGBRF batch-model list structure.

    Real structure: List[MultiOutputRegressor], each wrapping 1,000 XGB estimators.
    Here: 3 fake batches of 5 estimators each (total capacity = 15 genes).
    Each 'estimator' is a minimal sklearn-compatible object that can predict.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.multioutput import MultiOutputRegressor

    batch_size = 5
    n_batches = 3
    batches = []

    for _ in range(n_batches):
        # Train a tiny MultiOutputRegressor as a structural stand-in
        X_fake = RNG.standard_normal((20, N_TFS))
        y_fake = RNG.standard_normal((20, batch_size))
        mor = MultiOutputRegressor(LinearRegression())
        mor.fit(X_fake, y_fake)
        batches.append(mor)

    return batches, batch_size