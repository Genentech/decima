import pandas as pd
import h5py
import pytest
from click.testing import CliRunner
from decima.cli import main

from conftest import device


def test_cli_main():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0


@pytest.mark.long_running
def test_cli_download():
    runner = CliRunner()
    result = runner.invoke(main, ["download"])
    assert result.exit_code == 0


@pytest.mark.long_running
def test_cli_attributions_single_gene(tmp_path):
    output_dir = tmp_path / "SPI1"
    runner = CliRunner()
    result = runner.invoke(main, [
        "attributions",
        "--genes", "SPI1",
        "--tasks", "cell_type == 'classical monocyte'",
        "--output_dir", str(output_dir),
        "--model", "0",
        "--device", device,
        "--plot_seqlogo"
    ])
    assert result.exit_code == 0

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks_plots" / "SPI1.png").exists()
    assert (output_dir / "qc.warnings.log").exists()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "attributions.h5").exists()
    assert (output_dir / "seqlogos").is_dir()

    with h5py.File(output_dir / "attributions.h5", "r") as f:
        assert "SPI1" in f

    df_peaks = pd.read_csv(output_dir / "peaks.bed", sep="\t", header=None)
    genes = set(df_peaks[3].str.split("@").str[0])
    assert "SPI1" in genes

    df_motifs = pd.read_csv(output_dir / "motifs.tsv", sep="\t")
    genes = set(df_motifs['peak'].str.split("@").str[0])
    assert "SPI1" in genes

@pytest.mark.long_running
def test_cli_attributions_multiple_genes(tmp_path):
    output_dir = tmp_path / "SPI1_CD68_BRD3"
    runner = CliRunner()
    result = runner.invoke(main, [
        "attributions",
        "--genes", "SPI1,CD68,BRD3",
        "--tasks", "cell_type == 'classical monocyte'",
        "--output_dir", str(output_dir),
        "--model", "0",
        "--device", device
    ])
    assert result.exit_code == 0

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks_plots" / "SPI1.png").exists()
    assert (output_dir / "peaks_plots" / "CD68.png").exists()
    assert (output_dir / "peaks_plots" / "CD68.png").exists()
    assert (output_dir / "qc.warnings.log").exists()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "attributions.h5").exists()

    with open(output_dir / "qc.warnings.log", "r") as f:
        assert "BRD3" in f.read() # gene with low correlation

    with h5py.File(output_dir / "attributions.h5", "r") as f:
        assert "SPI1" in f
        assert "CD68" in f
        assert "BRD3" in f

    df_peaks = pd.read_csv(output_dir / "peaks.bed", sep="\t", header=None)
    genes = set(df_peaks[3].str.split("@").str[0])
    assert "SPI1" in genes
    assert "CD68" in genes
    assert "BRD3" in genes

    df_motifs = pd.read_csv(output_dir / "motifs.tsv", sep="\t")
    genes = set(df_motifs['peak'].str.split("@").str[0])
    assert "SPI1" in genes
    assert "CD68" in genes
    assert "BRD3" in genes

@pytest.mark.long_running
def test_cli_attributions_sequences(tmp_path):
    output_dir = tmp_path / "seqs"
    runner = CliRunner()
    result = runner.invoke(main, [
        "attributions",
        "--seqs", "tests/data/seqs.fasta",
        "--tasks", "cell_type == 'classical monocyte'",
        "--output_dir", str(output_dir),
        "--model", "0",
        "--device", device,
    ])
    assert result.exit_code == 0

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks_plots").is_dir()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "attributions.h5").exists()
    assert (output_dir / "seqs.fasta").exists()
    assert (output_dir / "seqs.fasta.fai").exists()

    with h5py.File(output_dir / "attributions.h5", "r") as f:
        assert "SPI1" in f
        assert "CD68" in f

    df_peaks = pd.read_csv(output_dir / "peaks.bed", sep="\t", header=None)
    genes = set(df_peaks[3].str.split("@").str[0])
    assert "CD68" in genes
    assert "SPI1" in genes

    df_motifs = pd.read_csv(output_dir / "motifs.tsv", sep="\t")
    genes = set(df_motifs['peak'].str.split("@").str[0])
    assert "CD68" in genes
    assert "SPI1" in genes


@pytest.mark.long_running
def test_cli_vep_tsv(tmp_path):
    output_file = tmp_path / "test_predictions.parquet"
    runner = CliRunner()
    result = runner.invoke(main, [
        "vep",
        "-v", "tests/data/variants.tsv",
        "-o", str(output_file),
        "--tasks", "cell_type == 'CD8-positive, alpha-beta T cell'",
        "--model", "0",
        "--device", device,
        "--max-distance", "20000",
        "--chunksize", "5"
    ])
    assert result.exit_code == 0

    assert output_file.exists()

    df_saved = pd.read_parquet(output_file)
    assert df_saved.shape[0] > 0  # Should have some predictions
    assert df_saved.shape[1] > 14  # annotationcolumns + prediction columns

    required_cols = ['chrom', 'pos', 'ref', 'alt', 'gene', 'start', 'end', 'strand',
                     'gene_mask_start', 'gene_mask_end', 'rel_pos', 'ref_tx', 'alt_tx', 'tss_dist']
    for col in required_cols:
        assert col in df_saved.columns


@pytest.mark.long_running
def test_cli_vep_vcf(tmp_path):
    output_file = tmp_path / "test_predictions.parquet"
    runner = CliRunner()
    result = runner.invoke(main, [
        "vep",
        "-v", "tests/data/test.vcf",
        "-o", str(output_file),
        "--tasks", "cell_type == 'CD8-positive, alpha-beta T cell'",
        "--model", "0",
        "--device", device,
        "--max-distance", "20000",
        "--chunksize", "5"
    ])
    assert result.exit_code == 0

    assert output_file.exists()

    df_saved = pd.read_parquet(output_file)
    assert df_saved.shape[0] > 0
    assert df_saved.shape[1] > 14

    required_cols = ['chrom', 'pos', 'ref', 'alt', 'gene', 'start', 'end', 'strand',
                     'gene_mask_start', 'gene_mask_end', 'rel_pos', 'ref_tx', 'alt_tx', 'tss_dist']
    for col in required_cols:
        assert col in df_saved.columns


@pytest.mark.long_running
def test_cli_vep_all_tasks(tmp_path):
    output_file = tmp_path / "test_predictions_all.parquet"
    runner = CliRunner()
    result = runner.invoke(main, [
        "vep",
        "-v", "tests/data/variants.tsv",
        "-o", str(output_file),
        "--model", "0",
        "--device", device,
        "--max-distance", "20000",
        "--chunksize", "5"
    ])
    assert result.exit_code == 0

    assert output_file.exists()

    df_saved = pd.read_parquet(output_file)
    assert df_saved.shape[0] > 0
    assert df_saved.shape[1] == 8870

    required_cols = ['chrom', 'pos', 'ref', 'alt', 'gene', 'start', 'end', 'strand',
                     'gene_mask_start', 'gene_mask_end', 'rel_pos', 'ref_tx', 'alt_tx', 'tss_dist']
    for col in required_cols:
        assert col in df_saved.columns


@pytest.mark.long_running
def test_cli_vep_all_tasks_ensemble_custom_genome(tmp_path):
    import genomepy
    output_file = tmp_path / "test_predictions_all_ensemble_custom_genome.parquet"
    runner = CliRunner()
    result = runner.invoke(main, [
        "vep",
        "-v", "tests/data/variants.tsv",
        "-o", str(output_file),
        "--model", "ensemble",
        "--device", device,
        "--max-distance", "20000",
        "--chunksize", "5",
        "--genome", genomepy.Genome("hg38").filename
    ])
    assert result.exit_code == 0

    assert output_file.exists()

    df_saved = pd.read_parquet(output_file)
    assert df_saved.shape[0] > 0
    assert df_saved.shape[1] == 8870


@pytest.mark.long_running
def test_cli_vep_all_tasks_ensemble(tmp_path):
    output_file = tmp_path / "test_predictions_all_ensemble.parquet"
    runner = CliRunner()
    result = runner.invoke(main, [
        "vep",
        "-v", "tests/data/variants.tsv",
        "-o", str(output_file),
        "--model", "ensemble",
        "--device", device,
        "--max-distance", "20000",
        "--chunksize", "5",
        "--save-replicates"
    ])
    assert result.exit_code == 0

    assert output_file.exists()

    df_saved = pd.read_parquet(output_file)
    assert df_saved.shape[0] > 0
    assert df_saved.shape[1] == 14 + 8856 * 5
