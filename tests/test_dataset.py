from decima.constants import DECIMA_CONTEXT_SIZE
from decima.data.dataset import GeneDataset


def test_gene_dataset():
    ds = GeneDataset()
    assert len(ds) == 18457
    assert ds[0].shape == (5, DECIMA_CONTEXT_SIZE)
