import numba
import pytest
import torch
import pyBigWig
import numpy as np
import pandas as pd
from grelu.sequence.format import strings_to_one_hot
from decima.interpret.attribution import Attribution
from decima import predict_save_attributions

from conftest import device


@pytest.fixture
def attributions():
    gene = "TEST2"
    input_seq = torch.zeros((4, 500))
    input_seq[:, 358:367] = strings_to_one_hot("CTGATAAGG") # GATA1 motif
    input_mask = torch.zeros((1, 500))
    input_mask[:, 350:375] = 1
    inputs = torch.vstack([input_seq, input_mask])

    np.random.seed(42)
    attrs = np.random.rand(4, 500) - .5
    attrs[:, 360:364] = 10_000

    return Attribution(
        gene=gene,
        inputs=inputs,
        attrs=attrs,
        chrom="chr1",
        start=1000,
        end=1500,
        gene_start=1350,
        gene_end=1375,
        strand="+",
        threshold=1e-2,
    )


def test_Attribution_peak_finding(attributions):
    assert len(attributions.peaks) == 1
    row = attributions.peaks.iloc[0]
    assert row["peak"] == "TEST2@10"
    assert row["start"] == 360
    assert row["end"] == 364
    assert row["attribution"] > 0
    assert row["p-value"] < 1e-2
    assert row["from_tss"] == 10


def test_Attribution_scan_motifs(attributions):
    df_motifs = attributions.scan_motifs()
    assert isinstance(df_motifs, pd.DataFrame)
    assert len(df_motifs) > 0

    assert "motif" in df_motifs.columns
    assert "p-value" in df_motifs.columns
    assert "peak" in df_motifs.columns
    assert "GATA1.H12CORE.1.PSM.A" in set(df_motifs["motif"])
    assert (df_motifs["peak"] == "TEST2@10").all()
    assert (df_motifs["start"] - 350 == df_motifs["from_tss"]).all()


def test_Attribution_peaks_to_bed(attributions):
    df_peaks = attributions.peaks_to_bed()
    assert isinstance(df_peaks, pd.DataFrame)
    assert len(df_peaks) == 1

    assert df_peaks.columns.tolist() == ["chrom", "start", "end", "name", "score", "strand"]

    row = df_peaks.iloc[0]
    assert row["chrom"] == "chr1"
    assert row["start"] == 1360
    assert row["end"] == 1364
    assert row["name"] == "TEST2@10"
    assert row["score"] > 2
    assert row["strand"] == "."


def test_Attribution_save_bigwig(attributions, tmp_path):
    bigwig_path = tmp_path / "test.bigwig"
    attributions.save_bigwig(str(bigwig_path))
    assert bigwig_path.exists()

    bw = pyBigWig.open(str(bigwig_path))
    attrs = bw.values("chr1", 1360, 1364)
    assert len(attrs) == 4
    assert attrs[0] == pytest.approx(40_000)
    bw.close()


def test_predict_save_attributions(tmp_path):
    output_dir = tmp_path / "SPI1"
    predict_save_attributions(str(output_dir), "SPI1", "cell_type == 'classical monocyte'", device=device)

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks.png").exists()
    assert (output_dir / "attributions.bigwig").exists()
    assert (output_dir / "attributions.npz").exists()
    assert (output_dir / "attributions_seq_logos").is_dir()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "qc.log").exists()
