"""
=============================
Unit tests for the REPO_ROOT/DATA_ROOT resolution infrastructure.

Every .py script in the repo uses a `Path(__file__).resolve().parents` walk
to locate the repo root. This is robust when the scripts are run from their
correct location, but has edge cases worth explicitly testing:

Weak chains targeted
--------------------
1. The README.md walk may find a parent repo's README.md if the project is
   nested (e.g., inside another git repo or a shared lab directory).
2. data_config.json parsing: malformed JSON, missing DATA_ROOT key, or a
   path that exists in the JSON but not on disk should produce clear
   errors, not cryptic KeyErrors or StopIteration.
3. The `if 'REPO_ROOT' not in dir()` guard prevents redundant I/O when
   scripts are %run from notebooks, but its semantics need to be understood
   correctly (dir() reflects the calling namespace, not globals()).
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Reference implementation of the path resolution logic
# ---------------------------------------------------------------------------

def resolve_repo_root(start_path: Path) -> Path:
    """
    Walk upward from start_path until a directory containing README.md
    is found. Raises RuntimeError if none is found.
    Mirrors the production pattern in all .py scripts.
    """
    for parent in start_path.resolve().parents:
        if (parent / "README.md").exists():
            return parent
    raise RuntimeError(
        "Could not find repo root: no README.md found in any parent directory."
    )


def load_data_config(repo_root: Path) -> str:
    """
    Load DATA_ROOT from data_config.json at repo_root.
    Returns the DATA_ROOT string value.
    Raises informative exceptions for all failure modes.
    """
    config_path = repo_root / "data_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"data_config.json not found at {config_path}. "
            "Create it with: {\"DATA_ROOT\": \"/path/to/your/data\"}"
        )
    with open(config_path) as f:
        try:
            cfg = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"data_config.json is not valid JSON: {exc}") from exc

    if "DATA_ROOT" not in cfg:
        raise KeyError(
            "data_config.json is missing the 'DATA_ROOT' key. "
            f"Found keys: {list(cfg.keys())}"
        )
    return cfg["DATA_ROOT"]


# ===========================================================================
# Tests: resolve_repo_root()
# ===========================================================================

class TestResolveRepoRoot:

    def test_finds_readme_in_parent(self, tmp_path):
        """Typical case: README.md is one level above the script."""
        (tmp_path / "README.md").touch()
        script_dir = tmp_path / "subdir"
        script_dir.mkdir()
        script_file = script_dir / "script.py"
        script_file.touch()

        root = resolve_repo_root(script_file)
        assert root == tmp_path

    def test_finds_readme_several_levels_up(self, tmp_path):
        """README.md two levels above the script (e.g. config/SHAP/script.py)."""
        (tmp_path / "README.md").touch()
        deep = tmp_path / "config" / "SHAP"
        deep.mkdir(parents=True)
        script = deep / "SHAP_generation_baseline.py"
        script.touch()

        root = resolve_repo_root(script)
        assert root == tmp_path

    def test_raises_when_no_readme_found(self, tmp_path):
        """No README.md in any ancestor → RuntimeError, not StopIteration."""
        script = tmp_path / "orphan_script.py"
        script.touch()
        with pytest.raises(RuntimeError, match="README.md"):
            resolve_repo_root(script)

    def test_finds_nearest_readme_not_furthest(self, tmp_path):
        """
        If README.md exists at multiple levels, the innermost one (closest
        ancestor) should be returned. This prevents attaching to an
        outer monorepo accidentally.
        """
        outer = tmp_path
        (outer / "README.md").touch()  # outer repo README

        inner = tmp_path / "LEMBAS-RNN-benchmark"
        inner.mkdir()
        (inner / "README.md").touch()  # correct repo README

        script_dir = inner / "config" / "predictions"
        script_dir.mkdir(parents=True)
        script = script_dir / "model_load.py"
        script.touch()

        root = resolve_repo_root(script)
        assert root == inner, (
            f"Expected innermost README at {inner}, got {root}. "
            "Outer monorepo README should not be used."
        )

    def test_actual_repo_root_is_resolved_correctly(self):
        """
        Integration smoke test: verify the actual repo root (found via conftest)
        contains the expected top-level markers.
        """
        from conftest import REPO_ROOT
        assert (REPO_ROOT / "README.md").exists()
        assert (REPO_ROOT / "config").exists()
        assert (REPO_ROOT / "run").exists()


# ===========================================================================
# Tests: load_data_config()
# ===========================================================================

class TestLoadDataConfig:

    def test_loads_valid_config(self, tmp_path):
        """Happy path: valid JSON with DATA_ROOT key."""
        config = {"DATA_ROOT": "/some/data/path"}
        (tmp_path / "data_config.json").write_text(json.dumps(config))
        result = load_data_config(tmp_path)
        assert result == "/some/data/path"

    def test_raises_file_not_found_when_config_missing(self, tmp_path):
        """Missing data_config.json must raise FileNotFoundError, not KeyError."""
        with pytest.raises(FileNotFoundError, match="data_config.json"):
            load_data_config(tmp_path)

    def test_raises_value_error_for_malformed_json(self, tmp_path):
        """Malformed JSON must raise ValueError with a useful message."""
        (tmp_path / "data_config.json").write_text("{DATA_ROOT: not valid json}")
        with pytest.raises(ValueError, match="not valid JSON"):
            load_data_config(tmp_path)

    def test_raises_key_error_when_data_root_missing(self, tmp_path):
        """JSON that doesn't contain DATA_ROOT must raise KeyError."""
        config = {"WRONG_KEY": "/some/path"}
        (tmp_path / "data_config.json").write_text(json.dumps(config))
        with pytest.raises(KeyError, match="DATA_ROOT"):
            load_data_config(tmp_path)

    def test_empty_json_object_raises_key_error(self, tmp_path):
        (tmp_path / "data_config.json").write_text("{}")
        with pytest.raises(KeyError):
            load_data_config(tmp_path)

    def test_accepts_path_with_spaces(self, tmp_path):
        """Data paths on Linux/Mac can contain spaces (e.g. 'Zhang Lab Data')."""
        config = {"DATA_ROOT": "/home/user/Zhang Lab Data"}
        (tmp_path / "data_config.json").write_text(json.dumps(config))
        result = load_data_config(tmp_path)
        assert result == "/home/user/Zhang Lab Data"

    def test_extra_keys_are_ignored(self, tmp_path):
        """Additional keys in the config file must not cause failures."""
        config = {"DATA_ROOT": "/valid/path", "EXTRA_KEY": "ignored"}
        (tmp_path / "data_config.json").write_text(json.dumps(config))
        result = load_data_config(tmp_path)
        assert result == "/valid/path"


# ===========================================================================
# Tests: sys.path idempotency (the if REPO_ROOT not in sys.path guard)
# ===========================================================================

class TestSysPathGuard:

    def test_repo_root_not_added_twice(self):
        """
        The production guard `if REPO_ROOT not in sys.path` prevents
        sys.path from accumulating duplicate entries across %run calls.
        Verify the guard logic works as expected.
        """
        from conftest import REPO_ROOT

        repo_str = str(REPO_ROOT)

        # Simulate the guard logic from every .py header
        before_count = sys.path.count(repo_str)

        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        after_count = sys.path.count(repo_str)

        # Should appear at most once regardless of initial state
        assert after_count == 1, (
            f"REPO_ROOT appears {after_count} times in sys.path; "
            "the guard should ensure it appears exactly once."
        )