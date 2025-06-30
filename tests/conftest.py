import os
import pandas as pd
import torch
import pytest
from decima.hub import login_wandb

import pytest

from decima.hub.download import download_hg38


def pytest_addoption(parser):
    """Adds a --run-long-running option to pytest."""
    parser.addoption(
        "--run-long-running", action="store_true", default=False, help="run tests marked as long_running"
    )

def pytest_configure(config):
    """Registers the 'long_running' marker."""
    config.addinivalue_line("markers", "long_running: mark test as long to run")

def pytest_collection_modifyitems(config, items):
    """
    Conditionally skips 'long_running' tests based on the --run-long-running option
    or the RUN_LONG_RUNNING_TESTS environment variable.
    """
    run_long_running = config.getoption("--run-long-running")
    env_var_set = os.getenv("RUN_LONG_RUNNING_TESTS", "false").lower() == "true"

    if not (run_long_running or env_var_set):
        # If neither the CLI option nor the environment variable is set to true, skip long_running tests
        skip_long_running = pytest.mark.skip(reason="need --run-long-running option or RUN_LONG_RUNNING_TESTS=true env var to run")
        for item in items:
            if "long_running" in item.keywords:
                item.add_marker(skip_long_running)


login_wandb()
download_hg38()

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"


@pytest.fixture
def df_variant():
    return pd.DataFrame({
        "chrom": ["chr1", "chr1", "chr1", "chr1", "chr1"],
        "pos": [1000018, 1002308, 109727471, 109728286, 109728807],
        "ref": ["G", "T", "A", "TTT", "T"],
        "alt": ["A", "C", "C", "G", "GG"],
    })
