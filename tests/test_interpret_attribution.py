import pytest
import torch
import pyBigWig
import numpy as np
import pandas as pd
from decima.interpret.attribution import Attribution
from decima import predict_save_attributions

from conftest import device

@pytest.fixture
def attributions():
    gene = "TEST2"
    inputs = torch.zeros((5, 100))
    inputs[-1, 50] = 1

    attrs = np.zeros((4, 100))
    attrs[:, 30] = 1.0
    attrs[:, 70] = 0.8

    return Attribution(
        gene=gene,
        inputs=inputs,
        attrs=attrs,
        chrom="chr1",
        start=1000,
        end=1100,
        gene_start=1050,
        gene_end=1075,
        strand="+",
        n_peaks=2,
        min_dist=5
    )


def test_Attribution_peak_finding(attributions):
    assert len(attributions.peaks) == 2
    assert attributions.peaks.iloc[0]["peak"] == 30
    assert attributions.peaks.iloc[1]["peak"] == 70
    assert "from_tss" in attributions.peaks.columns


def test_Attribution_scan_motifs(attributions):
    df_motifs = attributions.scan_motifs()
    assert isinstance(df_motifs, pd.DataFrame)
    assert len(df_motifs) > 0
    assert "motif" in df_motifs.columns
    assert "p-value" in df_motifs.columns
    assert "peak" in df_motifs.columns


def test_Attribution_peaks_to_bed(attributions):
    df_peaks = attributions.peaks_to_bed()
    assert isinstance(df_peaks, pd.DataFrame)
    assert len(df_peaks) == 2
    assert df_peaks.columns.tolist() == ["chrom", "start", "end", "name", "score", "strand"]


def test_Attribution_save_bigwig(attributions, tmp_path):
    bigwig_path = tmp_path / "test.bigwig"
    attributions.save_bigwig(str(bigwig_path))
    assert bigwig_path.exists()

    bw = pyBigWig.open(str(bigwig_path))
    attrs = bw.values("chr1", 1000, 1100)
    assert len(attrs) == 100
    assert attrs[30] == pytest.approx(1.0 * 4)
    assert attrs[70] == pytest.approx(0.8 * 4)
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
