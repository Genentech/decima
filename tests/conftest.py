import os
import numpy as np
import pandas as pd
import h5py
import torch
import pytest

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.hub import login_wandb
from decima.hub.download import cache_hg38


fasta_file = "tests/data/seqs.fasta"

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
cache_hg38()


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



@pytest.fixture
def attribution_data():
    np.random.seed(42)

    genes = ['PDIA3', 'EIF2S3', 'PCNP', 'SELENOT', 'DNAJA1', 'TFAM', 'RSL24D1', 'PSMB7', 'ATP6V1E1', 'NRBP1']

    sequences = np.random.randint(0, 4, (len(genes), DECIMA_CONTEXT_SIZE)).astype('i1')
    attributions = np.random.randn(len(genes), 4, DECIMA_CONTEXT_SIZE).astype(np.float32)

    return {
        'genes': genes,
        'sequences': sequences,
        'attributions': attributions,
        'gene_mask_start': [163_840] * len(genes),
        'gene_mask_end': [223_490] * len(genes)
    }


@pytest.fixture
def attribution_h5_file(tmp_path, attribution_data):
    h5_path = tmp_path / "test_attributions.h5"

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('genes', data=[name.encode('utf-8') for name in attribution_data['genes']])
        f.create_dataset('sequence', data=attribution_data['sequences'])
        f.create_dataset('attribution', data=attribution_data['attributions'])
        f.create_dataset('gene_mask_start', data=attribution_data['gene_mask_start'])
        f.create_dataset('gene_mask_end', data=attribution_data['gene_mask_end'])
        f.attrs['model_name'] = 'v1_rep0'
        f.attrs['genome'] = 'hg38'

    return h5_path
