import pytest
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

from decima.core.attribution import AttributionResult
from decima.constants import DECIMA_CONTEXT_SIZE


@pytest.fixture
def attribution_data():
    np.random.seed(42)

    genes = ['PDIA3', 'EIF2S3', 'PCNP', 'SELENOT', 'DNAJA1', 'TFAM', 'RSL24D1', 'PSMB7', 'ATP6V1E1', 'NRBP1']

    sequences = np.random.randint(0, 2, (len(genes), 4, DECIMA_CONTEXT_SIZE)).astype('i1')
    for i in range(len(genes)):
        for j in range(DECIMA_CONTEXT_SIZE):
            sequences[i, :, j] = 0
            hot_idx = np.random.randint(0, 4)
            sequences[i, hot_idx, j] = 1

    attributions = np.random.randn(len(genes), 4, DECIMA_CONTEXT_SIZE).astype(np.float32)

    return {
        'genes': genes,
        'sequences': sequences,
        'attributions': attributions
    }


@pytest.fixture
def attribution_h5_file(tmp_path, attribution_data):
    h5_path = tmp_path / "test_attributions.h5"

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('genes', data=[name.encode('utf-8') for name in attribution_data['genes']])
        f.create_dataset('sequence', data=attribution_data['sequences'])
        f.create_dataset('attribution', data=attribution_data['attributions'])
        f.attrs['model_name'] = 'test_model'
        f.attrs['genome'] = 'hg38'

    return h5_path


def test_AttributionResult(attribution_h5_file, attribution_data):
    with AttributionResult(str(attribution_h5_file), tss_distance=10_000) as ar:
        assert len(ar.genes) == 10
        assert ar.genes == attribution_data['genes']
        assert ar.model_name == 'test_model'
        assert ar.genome == 'hg38'

        assert ar.genes == attribution_data['genes']
        genes = attribution_data['genes']
        seqs, attrs = ar.load(genes)

        assert seqs.shape == (10, 4, 20_000)
        assert attrs.shape == (10, 4, 20_000)

        start, end = 163_840 - 10_000, 163_840 + 10_000
        assert np.allclose(seqs, attribution_data['sequences'][:, :, start:end].astype(np.float32))

    with AttributionResult(str(attribution_h5_file), tss_distance=None) as ar:

        seqs, attrs = ar.load(genes[:1])
        assert seqs.shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert attrs.shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert np.allclose(seqs, attribution_data['sequences'][:1].astype(np.float32))

        seqs, attrs = ar.load(genes[8:9])
        assert seqs.shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert attrs.shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert np.allclose(seqs, attribution_data['sequences'][8:9].astype(np.float32))
