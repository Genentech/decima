import pandas as pd
import torch
from grelu.sequence.format import strings_to_one_hot

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.data.dataset import GeneDataset, SeqDataset


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


def test_SeqDataset():
    dataset = SeqDataset(
        seqs=["A" * DECIMA_CONTEXT_SIZE, "T" * DECIMA_CONTEXT_SIZE, "C" * DECIMA_CONTEXT_SIZE],
        gene_mask_starts=[1, 1, 1],
        gene_mask_ends=[2, 2, 2],
    )
    assert len(dataset) == 3
    assert dataset[0].shape == (5, DECIMA_CONTEXT_SIZE)
    assert dataset[1].shape == (5, DECIMA_CONTEXT_SIZE)

    df = pd.DataFrame(
        {
            "seq": ["A" * DECIMA_CONTEXT_SIZE, "T" * DECIMA_CONTEXT_SIZE, "C" * DECIMA_CONTEXT_SIZE],
            "gene_mask_start": [1, 1, 1],
            "gene_mask_end": [2, 2, 2],
        }
    )
    dataset = SeqDataset.from_dataframe(df)
    assert len(dataset) == 3
    assert dataset[0].shape == (5, DECIMA_CONTEXT_SIZE)

    dataset = SeqDataset.from_fasta("tests/data/seqs.fasta")
    assert len(dataset) == 2
    assert dataset[0].shape == (5, DECIMA_CONTEXT_SIZE)

    one_hot = torch.cat([
        strings_to_one_hot(["A" * DECIMA_CONTEXT_SIZE, "T" * DECIMA_CONTEXT_SIZE]),
        torch.ones(2, 1, DECIMA_CONTEXT_SIZE),
    ], dim=1)
    dataset = SeqDataset.from_one_hot(one_hot)
    assert len(dataset) == 2
    assert dataset[0].shape == (5, DECIMA_CONTEXT_SIZE)
    assert dataset.gene_mask_starts == [0, 0]
    assert dataset.gene_mask_ends == [524287, 524287]
