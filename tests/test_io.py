import numpy as np
import h5py
import torch
import pytest
from grelu.sequence.format import convert_input_type

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

    with AttributionWriter(str(h5_file), genes, "v1_rep0", bigwig=False) as writer:
        writer.add("STRADA", seqs, attrs)

    assert h5_file.exists()

    with h5py.File(h5_file, "r") as f:
        assert set(f.keys()) == {"genes", "attribution", "sequence", "gene_mask_start", "gene_mask_end"}
        assert f.attrs["model_name"] == "v1_rep0"
        assert f["attribution"].shape == (1, 4, DECIMA_CONTEXT_SIZE)
        np.testing.assert_array_equal(convert_input_type(f["sequence"][0], "one_hot", input_type="indices"), seqs)
        np.testing.assert_array_almost_equal(f["attribution"][0], attrs)
        assert [g.decode('utf-8') for g in f["genes"][:]] == ["STRADA"]
        assert f["gene_mask_start"][0] == 163840
        assert f["gene_mask_end"][0] == 223490

    h5_file = tmp_path / "test_bigwig.h5"
    with AttributionWriter(str(h5_file), genes, "v1_rep0", bigwig=True) as writer:
        writer.add("STRADA", seqs, attrs)

    assert h5_file.with_suffix(".bigwig").exists()

    h5_file = tmp_path / "test_bigwig_custom.h5"
    with AttributionWriter(str(h5_file), genes, "v1_rep0", bigwig=True, custom_genes=True) as writer:
        writer.add("STRADA", seqs, attrs, gene_mask_start=100, gene_mask_end=200)

    with h5py.File(h5_file, "r") as f:
        assert f["gene_mask_start"][0] == 100
        assert f["gene_mask_end"][0] == 200
