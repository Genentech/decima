import numpy as np
import h5py
import pytest

from decima.utils.io import read_fasta_gene_mask, BigWigWriter, AttributionWriter
from decima.constants import DECIMA_CONTEXT_SIZE


def test_read_fasta():
    df = read_fasta_gene_mask("tests/data/seqs.fasta")
    assert df.columns.tolist() == ['seq', 'gene_mask_start', 'gene_mask_end']
    assert df.index.tolist() == ['CD68', 'SPI1']


def test_BigWigWriter(tmp_path):
    bigwig_file = tmp_path / "test.bigwig"

    with BigWigWriter(str(bigwig_file), "hg38") as writer:
        chrom, start, end = "chr17", 63381538, 63381638
        values = np.random.rand(100)
        writer.add(chrom, start, end, values)

    assert bigwig_file.exists()
    assert bigwig_file.stat().st_size > 0


def test_AttributionWriter(tmp_path):
    genes = ["STRADA"]
    h5_file = tmp_path / "test.h5"
    attrs = np.random.rand(4, DECIMA_CONTEXT_SIZE)

    seqs = np.zeros((4, DECIMA_CONTEXT_SIZE))
    nucleotide_indices = np.random.randint(0, 4, DECIMA_CONTEXT_SIZE)
    seqs[nucleotide_indices, np.arange(DECIMA_CONTEXT_SIZE)] = 1

    with AttributionWriter(str(h5_file), genes, "test_model", bigwig=False) as writer:
        writer.add("STRADA", seqs, attrs)

    assert h5_file.exists()

    with h5py.File(h5_file, "r") as f:
        assert set(f.keys()) == {"genes", "attribution", "sequence"}
        assert f.attrs["model_name"] == "test_model"
        assert f["attribution"].shape == (1, 4, DECIMA_CONTEXT_SIZE)
        np.testing.assert_array_equal(f["sequence"][0], seqs)
        np.testing.assert_array_almost_equal(f["attribution"][0], attrs)
        assert [g.decode('utf-8') for g in f["genes"][:]] == ["STRADA"]

    h5_file = tmp_path / "test_bigwig.h5"
    with AttributionWriter(str(h5_file), genes, "test_model", bigwig=True) as writer:
        writer.add("STRADA", seqs, attrs)

    assert h5_file.with_suffix(".bigwig").exists()
