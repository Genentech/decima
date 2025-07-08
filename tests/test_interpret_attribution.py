import numba
import pytest
import torch
import pyBigWig
import numpy as np
import pandas as pd
from grelu.sequence.format import strings_to_one_hot
from captum.attr import Saliency, InputXGradient, IntegratedGradients
from decima.interpret.attributions import Attribution
from decima import predict_save_attributions
from decima.interpret.attributions import get_attribution_method
from decima.constants import DECIMA_CONTEXT_SIZE

from conftest import device


def test_get_attribution_method():
    assert get_attribution_method("saliency") == Saliency
    assert get_attribution_method("inputxgradient") == InputXGradient
    assert get_attribution_method("integratedgradients") == IntegratedGradients

@pytest.fixture
def attributions():
    gene = "TEST2"
    input_seq = torch.zeros((4, 500))
    input_seq[:, 358:367] = strings_to_one_hot("CTGATAAGG") # GATA1 motif
    input_mask = torch.zeros((1, 500))
    input_mask[:, 350:375] = 1
    inputs = torch.vstack([input_seq, input_mask])

    np.random.seed(42)
    attrs = np.random.rand(4, 500) / 10 - 0.05
    attrs[:, 360:364] = 10
    return Attribution(
        gene=gene,
        inputs=inputs,
        attrs=attrs,
        chrom="chr1",
        start=1000,
        end=1500,
        strand="+",
        threshold=1e-3,
    )


def test_Attribution_peak_finding(attributions):
    assert len(attributions.peaks) == 2
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
    assert "TEST2@10" in set(df_motifs["peak"])

    df_motifs = df_motifs[df_motifs["peak"] == "TEST2@10"]
    assert (df_motifs["start"] - 350 == df_motifs["from_tss"]).all()


def test_Attribution_peaks_to_bed(attributions):
    df_peaks = attributions.peaks_to_bed()
    assert isinstance(df_peaks, pd.DataFrame)
    assert len(df_peaks) == 2

    assert df_peaks.columns.tolist() == ["chrom", "start", "end", "name", "score", "strand"]

    row = df_peaks.iloc[0]
    assert row["chrom"] == "chr1"
    assert row["start"] == 1360
    assert row["end"] == 1364
    assert row["name"] == "TEST2@10"
    assert row["score"] > 2
    assert row["strand"] == "."

    attributions._strand = "-"
    df_peaks = attributions.peaks_to_bed()
    assert df_peaks.iloc[0]["strand"] == "."
    assert row["start"] == 1360
    assert row["end"] == 1364


@pytest.mark.long_running
def test_Attribution_from_seq(tmp_path):
    attributions = Attribution.from_seq(
        inputs='A' * DECIMA_CONTEXT_SIZE,
        gene_mask_start=0,
        gene_mask_end=DECIMA_CONTEXT_SIZE,
    )
    bigwig_path = tmp_path / "test.bigwig"
    attributions.save_bigwig(str(bigwig_path))
    bw = pyBigWig.open(str(bigwig_path))
    attrs = bw.values("custom", 0, DECIMA_CONTEXT_SIZE)
    assert len(attrs) == DECIMA_CONTEXT_SIZE
    assert np.sum(attrs) < 1 # no expression for AAAAAAAAAA...
    bw.close()


def test_Attribution_save_bigwig(attributions, tmp_path):
    bigwig_path = tmp_path / "test.bigwig"
    attributions.save_bigwig(str(bigwig_path))
    assert bigwig_path.exists()

    bw = pyBigWig.open(str(bigwig_path))
    attrs = bw.values("chr1", 1360, 1364)
    assert len(attrs) == 4
    assert attrs[0] == pytest.approx(40)
    bw.close()

@pytest.mark.long_running
def test_predict_save_attributions_single_gene(tmp_path):
    output_dir = tmp_path / "SPI1"
    predict_save_attributions(output_dir=str(output_dir), genes=["SPI1"], tasks="cell_type == 'classical monocyte'", device=device)

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks_plots").is_dir()
    assert (output_dir / "attributions.h5").exists()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "qc.warnings.log").exists()


@pytest.mark.long_running
def test_predict_save_attributions_single_gene_saliency(tmp_path):
    output_dir = tmp_path / "SPI1"
    predict_save_attributions(output_dir=str(output_dir), genes=["SPI1"], method="saliency", tasks="cell_type == 'classical monocyte'", device=device)

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks_plots").is_dir()
    assert (output_dir / "attributions.h5").exists()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "qc.warnings.log").exists()


@pytest.mark.long_running
def test_predict_save_attributions_single_gene_inputxgradient(tmp_path):
    output_dir = tmp_path / "SPI1"
    predict_save_attributions(output_dir=str(output_dir), genes=["SPI1"], method="inputxgradient", tasks="cell_type == 'classical monocyte'", device=device)

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks_plots").is_dir()
    assert (output_dir / "attributions.h5").exists()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "qc.warnings.log").exists()


@pytest.mark.long_running
def test_predict_save_attributions_single_gene_integratedgradients(tmp_path):
    output_dir = tmp_path / "SPI1"
    predict_save_attributions(output_dir=str(output_dir), genes=["SPI1"], method="integratedgradients", tasks="cell_type == 'classical monocyte'", device=device)

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks_plots").is_dir()
    assert (output_dir / "attributions.h5").exists()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "qc.warnings.log").exists()


@pytest.mark.long_running
def test_predict_save_attributions_multiple_genes(tmp_path):
    output_dir = tmp_path / "SPI1_CD68"
    predict_save_attributions(output_dir=str(output_dir), genes=["SPI1", "CD68"], tasks="cell_type == 'classical monocyte'", device=device)

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks_plots").is_dir()
    assert (output_dir / "attributions.h5").exists()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "qc.warnings.log").exists()


@pytest.mark.long_running
def test_predict_save_attributions_seqs(tmp_path):
    output_dir = tmp_path / "seqs"
    seqs = pd.read_csv('tests/data/seqs.csv', index_col=0)
    predict_save_attributions(output_dir=str(output_dir), seqs=seqs, tasks="cell_type == 'classical monocyte'", device=device)
