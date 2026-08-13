from pathlib import Path

import matplotlib
import pytest

# Headless/non-interactive backend for the whole test session so chart
# tests never try to open a GUI window in CI.
matplotlib.use("Agg")


@pytest.fixture
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent
