from decima.constants import DECIMA_CONTEXT_SIZE
from decima.data.dataset import GeneDataset


def test_gene_dataset():
    ds = GeneDataset()
    assert len(ds) == 18457
    assert ds[0].shape == (5, DECIMA_CONTEXT_SIZE)

    ds = GeneDataset(genes=[
        "SPI1", "GATA1", "ARMC5", "CD68", "FOXN3", "SOX3", "COL11A2", "JUN", "UBXN2A", "COX5B"
    ])
    assert len(ds) == 10
    assert ds[0].shape == (5, DECIMA_CONTEXT_SIZE)

    ds = GeneDataset(max_seq_shift=100)
    assert len(ds) == 18457

    ds = GeneDataset(max_seq_shift=100, augment_mode="serial")
    assert len(ds) == 18457 * 201
