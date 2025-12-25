from pathlib import Path
import h5py
import numba
import pytest
import torch
import pyBigWig
import numpy as np
import pandas as pd
from grelu.sequence.format import strings_to_one_hot

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.hub.download import download_decima_weights, download_decima_metadata
from captum.attr import Saliency, InputXGradient, IntegratedGradients
from decima.core.attribution import Attribution
from decima import predict_attributions_seqlet_calling
from decima.interpret.attributer import DecimaAttributer, get_attribution_method
from decima.interpret.attributions import predict_save_attributions, recursive_seqlet_calling, plot_attributions

from conftest import device, fasta_file, attribution_h5_file


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
    attrs[:, 260:264] = -1
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
    assert row["peak"] == "pos.TEST2@8"
    assert row["start"] == 358
    assert row["end"] == 366
    assert row["attribution"] > 0
    assert row["p-value"] < 1e-2
    assert row["from_tss"] == 8

    row = attributions.peaks.iloc[1]
    assert row["peak"] == "neg.TEST2@-92"
    assert row["start"] == 258
    assert row["end"] == 266
    assert row["attribution"] < 0
    assert row["p-value"] < 1e-2
    assert row["from_tss"] == -92


def test_Attribution_scan_motifs(attributions):
    attrs = attributions._get_attrs(-10, 20)
    assert attrs.shape == (4, 30)
    assert np.allclose(attrs[:, :10], 0)

    attrs = attributions._get_attrs(360, 370)
    assert attrs.shape == (4, 10)
    assert np.allclose(attrs[:, :4], 10)

    attrs = attributions._get_attrs(490, 510)
    assert attrs.shape == (4, 20)
    assert np.allclose(attrs[:, 10:], 0)

    df_motifs = attributions.scan_motifs()
    assert isinstance(df_motifs, pd.DataFrame)
    assert len(df_motifs) > 0

    assert "motif" in df_motifs.columns
    assert "p-value" in df_motifs.columns
    assert "peak" in df_motifs.columns
    assert 'GATA1.H13CORE.0.P.B' in set(df_motifs["motif"])
    assert {'neg.TEST2@-92', 'pos.TEST2@8'} == set(df_motifs["peak"])

    df_motifs = df_motifs[df_motifs["peak"] == "pos.TEST2@8"]
    assert (df_motifs["start"] - 350 == df_motifs["from_tss"]).all()


def test_Attribution_peaks_to_bed(attributions):
    df_peaks = attributions.peaks_to_bed()
    assert isinstance(df_peaks, pd.DataFrame)
    assert len(df_peaks) == 2

    assert df_peaks.columns.tolist() == ["chrom", "start", "end", "name", "score", "strand", "attribution"]

    row = df_peaks.iloc[1]
    assert row["chrom"] == "chr1"
    assert row["start"] == 1358
    assert row["end"] == 1366
    assert row["name"] == "pos.TEST2@8"
    assert row["score"] > 2
    assert row["strand"] == "."

    attributions._strand = "-"
    df_peaks = attributions.peaks_to_bed()
    assert df_peaks.iloc[0]["strand"] == "."
    assert row["start"] == 1358
    assert row["end"] == 1366


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
    attrs = bw.values("chr1", 1358, 1366)
    assert len(attrs) == 8
    assert attrs[2:6] == [40, 40, 40, 40]
    bw.close()


@pytest.mark.long_running
def test__predict_save_attributions(tmp_path):
    output_prefix = tmp_path / "test"
    predict_save_attributions(
        output_prefix=output_prefix,
        tasks="(cell_type == 'classical monocyte') and (tissue == 'blood')",
        off_tasks="(cell_type != 'classical monocyte') and (tissue == 'blood')",
        top_n_markers=5,
        model=0,
        device=device,
    )
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    attribution_file = tmp_path / "test.attributions.h5"
    with h5py.File(attribution_file, "r") as f:
        assert f["attribution"].shape == (5, 4, DECIMA_CONTEXT_SIZE)
        assert f["sequence"].shape == (5, DECIMA_CONTEXT_SIZE)
        assert list(f["genes"][:]) == [b'MEFV', b'AQP9', b'CLEC5A', b'CLEC4D', b'PLA2G7']
        assert list(f["gene_mask_start"][:]) == [163840, 163840, 163840, 163840, 163840]
        assert list(f["gene_mask_end"][:]) == [178439, 211582, 183490, 176731, 195332]
        assert f.attrs['model_name'] == 'v1_rep0'


@pytest.mark.long_running
def test__predict_save_attributions_seqs(tmp_path):
    output_prefix = tmp_path / "test"
    predict_save_attributions(
        output_prefix=output_prefix,
        seqs=fasta_file,
        model=0,
        device=device,
    )
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    attribution_file = tmp_path / "test.attributions.h5"
    with h5py.File(attribution_file, "r") as f:
        assert list(f["genes"][:]) == [b"CD68", b"SPI1"]
        assert f["attribution"].shape == (2, 4, DECIMA_CONTEXT_SIZE)
        assert f["sequence"].shape == (2, DECIMA_CONTEXT_SIZE)
        assert list(f["gene_mask_start"][:]) == [163840, 163840]
        assert list(f["gene_mask_end"][:]) == [166460, 187556]
        assert f.attrs['model_name'] == 'v1_rep0'


@pytest.mark.long_running
def test_predict_save_attributions_single_gene(tmp_path):
    output_prefix = tmp_path / "SPI1"
    predict_attributions_seqlet_calling(
        output_prefix=output_prefix,
        genes=["SPI1"],
        tasks="cell_type == 'classical monocyte'",
        model=0,
        device=device
    )
    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    plot_attributions(output_prefix=output_prefix, genes=["SPI1"])
    plot_dir = Path(str(output_prefix) + "_plots")
    assert (plot_dir / "SPI1.peaks.png").exists()
    assert ((plot_dir / "SPI1_seqlogos").is_dir())


@pytest.mark.long_running
def test_predict_save_attributions_single_gene_list_models(tmp_path):
    # download models
    download_decima_weights("v1_rep0", str(tmp_path))
    download_decima_weights("v1_rep1", str(tmp_path))
    download_decima_metadata("v1_rep0", str(tmp_path))

    output_prefix = tmp_path / "SPI1"
    predict_attributions_seqlet_calling(
        output_prefix=output_prefix,
        genes=["SPI1"],
        metadata_anndata=str(tmp_path / "metadata.h5ad"),
        tasks="cell_type == 'classical monocyte'",
        model=[
            str(tmp_path / "rep0.safetensors"),
            str(tmp_path / "rep1.safetensors"),
        ],
        device=device
    )
    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert Path(str(output_prefix) + "_0.attributions.h5").exists()
    assert Path(str(output_prefix) + "_1.attributions.h5").exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert Path(str(output_prefix) + "_0.warnings.qc.log").exists()
    assert Path(str(output_prefix) + "_1.warnings.qc.log").exists()


@pytest.mark.long_running
def test_predict_save_attributions_single_gene_saliency(tmp_path):
    output_prefix = tmp_path / "SPI1"
    predict_attributions_seqlet_calling(
        output_prefix=output_prefix,
        genes=["SPI1"],
        method="saliency",
        tasks="cell_type == 'classical monocyte'",
        model=0,
        device=device
    )
    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    plot_attributions(output_prefix=output_prefix, genes=["SPI1"])
    plot_dir = Path(str(output_prefix) + "_plots")
    assert (plot_dir / "SPI1.peaks.png").exists()
    assert ((plot_dir / "SPI1_seqlogos").is_dir())


@pytest.mark.long_running
def test_predict_save_attributions_single_gene_inputxgradient(tmp_path):
    output_prefix = tmp_path / "SPI1"
    predict_attributions_seqlet_calling(
        output_prefix=output_prefix,
        genes=["SPI1"],
        method="inputxgradient",
        tasks="cell_type == 'classical monocyte'",
        model=0,
        device=device
    )
    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    plot_attributions(output_prefix=output_prefix, genes=["SPI1"])
    plot_dir = Path(str(output_prefix) + "_plots")
    assert (plot_dir / "SPI1.peaks.png").exists()
    assert ((plot_dir / "SPI1_seqlogos").is_dir())


@pytest.mark.long_running
def test_predict_save_attributions_single_gene_integratedgradients(tmp_path):
    output_prefix = tmp_path / "SPI1"
    predict_attributions_seqlet_calling(
        output_prefix=output_prefix,
        genes=["SPI1"],
        method="integratedgradients",
        tasks="cell_type == 'classical monocyte'",
        model=0,
        device=device
    )
    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    plot_attributions(output_prefix=output_prefix, genes=["SPI1"])
    plot_dir = Path(str(output_prefix) + "_plots")
    assert (plot_dir / "SPI1.peaks.png").exists()
    assert ((plot_dir / "SPI1_seqlogos").is_dir())


@pytest.mark.long_running
def test_predict_save_attributions_multiple_genes(tmp_path):
    output_prefix = tmp_path / "SPI1_CD68"
    predict_attributions_seqlet_calling(
        output_prefix=output_prefix,
        genes=["SPI1", "CD68"],
        tasks="cell_type == 'classical monocyte'",
        model=0,
        device=device
    )
    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    plot_attributions(output_prefix=output_prefix, genes=["CD68"])
    plot_dir = Path(str(output_prefix) + "_plots")
    assert (plot_dir / "CD68.peaks.png").exists()
    assert ((plot_dir / "CD68_seqlogos").is_dir())


@pytest.mark.long_running
def test_predict_save_attributions_seqs(tmp_path):
    output_prefix = tmp_path / "seqs"
    seqs = pd.read_csv('tests/data/seqs.csv', index_col=0)
    predict_attributions_seqlet_calling(
        output_prefix=output_prefix,
        seqs=seqs,
        tasks="cell_type == 'classical monocyte'",
        device=device,
        model=0,
        num_workers=1,
    )
    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert (output_prefix.with_suffix(".seqs.fasta")).exists()
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    plot_attributions(output_prefix=output_prefix, genes=["CD68"])
    plot_dir = Path(str(output_prefix) + "_plots")
    assert (plot_dir / "CD68.peaks.png").exists()
    assert ((plot_dir / "CD68_seqlogos").is_dir())


@pytest.mark.long_running
def test_DecimaAttributer():
    attributer = DecimaAttributer.load_decima_attributer(
        model_name=0,
        tasks=["agg_0", "agg_1", "agg_2"],
        method="inputxgradient",
        device=device
    )
    assert attributer.method == "inputxgradient"
    assert attributer.transform == "specificity"
    assert attributer.model is not None

    batch_size = 1
    dna_seq = 'A' * DECIMA_CONTEXT_SIZE
    one_hot_seq = strings_to_one_hot(dna_seq).unsqueeze(0)
    gene_mask = torch.zeros(batch_size, 1, DECIMA_CONTEXT_SIZE)
    gene_mask[:, :, 150_000:151_000] = 1
    inputs = torch.cat([one_hot_seq, gene_mask], dim=1).to(device)

    attrs = attributer.attribute(inputs)

    assert attrs.shape == (batch_size, 4, DECIMA_CONTEXT_SIZE)
    assert attrs.dtype == torch.float32

    with pytest.raises(ValueError):
        attributer = DecimaAttributer.load_decima_attributer(
            model_name=0,
            tasks=["agg_0", "agg_1", "agg_2"],
            transform="aggregate",
            off_tasks=["agg_4", "agg_5", "agg_6"],
            device=device
        )

    with pytest.warns(UserWarning):
        attributer = DecimaAttributer.load_decima_attributer(
            model_name=0,
            tasks=["agg_0", "agg_1", "agg_2"],
            transform="specificity",
            off_tasks=None,
            device=device
        )


@pytest.mark.long_running
def test_recursive_seqlet_calling(tmp_path, attribution_h5_file):
    output_prefix = tmp_path / "test"
    recursive_seqlet_calling(
        output_prefix=output_prefix,
        attributions=attribution_h5_file,
        genes=['PDIA3', 'EIF2S3', 'PCNP', 'SELENOT'],
        agg_func="mean"
    )
    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()


def test_Attribution_sub(attributions):
    # Test valid subtraction
    other_attrs = attributions.attrs * 2
    other = Attribution(
        gene="OTHER_GENE",
        inputs=attributions.inputs,
        attrs=other_attrs,
        chrom=attributions.chrom,
        start=attributions.start,
        end=attributions.end,
        strand=attributions.strand,
        threshold=attributions.threshold,
    )

    diff = other - attributions
    assert (diff.attrs == attributions.attrs).all()
    assert diff.gene == "OTHER_GENE-TEST2"
    assert diff.chrom == attributions.chrom
    assert diff.start == attributions.start

    # Test failure: different metadata
    other_diff_chrom = Attribution(
        gene="OTHER_GENE",
        inputs=attributions.inputs,
        attrs=other_attrs,
        chrom="chr2",
        start=attributions.start,
        end=attributions.end,
        strand=attributions.strand,
        threshold=attributions.threshold,
    )
    with pytest.raises(AssertionError, match="Chromosomes must be the same"):
        _ = other_diff_chrom - attributions

def test_Attribution__sub__(attributions):
    other = Attribution(
        gene="OTHER",
        inputs=attributions.inputs,
        attrs=attributions.attrs + 1,
        chrom=attributions.chrom,
        start=attributions.start,
        end=attributions.end,
        strand=attributions.strand,
        threshold=attributions.threshold,
    )

    diff = other - attributions
    assert np.allclose(diff.attrs, 1.0)
    assert diff.gene == "OTHER-TEST2"
