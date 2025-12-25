from pathlib import Path
import pandas as pd
import h5py
import pytest
from click.testing import CliRunner
from decima.cli import main

from conftest import device
from decima.constants import DECIMA_CONTEXT_SIZE, DEFAULT_ENSEMBLE


def test_cli_main():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0


@pytest.mark.long_running
def test_cli_cache():
    runner = CliRunner()
    result = runner.invoke(main, ["cache"])
    assert result.exit_code == 0


@pytest.mark.long_running
def test_cli_download(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["download", "--download-dir", str(tmp_path)])
    assert result.exit_code == 0


@pytest.mark.long_running
def test_cli_download_weights(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["download-weights", "--download-dir", str(tmp_path)])
    assert result.exit_code == 0


@pytest.mark.long_running
def test_cli_download_metadata(tmp_path):
    runner = CliRunner()
    result = runner.invoke(main, ["download-metadata", "--download-dir", str(tmp_path)])
    assert result.exit_code == 0


@pytest.mark.long_running
def test_cli_attributions_single_gene(tmp_path):
    output_prefix = tmp_path / "SPI1"
    runner = CliRunner()
    result = runner.invoke(main, [
        "attributions",
        "--genes", "SPI1",
        "--tasks", "cell_type == 'classical monocyte'",
        "--output-prefix", str(output_prefix),
        "--model", "0",
        "--device", device,
    ])
    assert result.exit_code == 0

    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".attributions.bigwig")).exists()
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    with h5py.File(output_prefix.with_suffix(".attributions.h5"), "r") as f:
        assert list(f.keys()) == ['attribution', 'gene_mask_end', 'gene_mask_start', 'genes', 'sequence']
        assert f["genes"][:]== [b"SPI1"]
        assert f["attribution"].shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert f["sequence"].shape == (1, DECIMA_CONTEXT_SIZE)
        assert f.attrs["model_name"] == "v1_rep0"
        assert f.attrs["genome"] == "hg38"

    df_peaks = pd.read_csv(output_prefix.with_suffix(".seqlets.bed"), sep="\t", header=None)
    genes = set(df_peaks[3].str.split(".").str[1].str.split("@").str[0])
    assert "SPI1" in genes

    df_motifs = pd.read_csv(output_prefix.with_suffix(".motifs.tsv"), sep="\t")
    genes = set(df_motifs['peak'].str.split(".").str[1].str.split("@").str[0])
    assert "SPI1" in genes


@pytest.mark.long_running
def test_cli_attributions_multiple_genes(tmp_path):
    output_prefix = tmp_path / "SPI1_CD68_BRD3"
    runner = CliRunner()
    result = runner.invoke(main, [
        "attributions",
        "--genes", "SPI1,CD68,BRD3",
        "--tasks", "cell_type == 'classical monocyte'",
        "--output-prefix", str(output_prefix),
        "--model", "0",
        "--device", device
    ])
    assert result.exit_code == 0

    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".attributions.bigwig")).exists()
    assert (output_prefix.with_suffix(".warnings.qc.log")).exists()

    with open(output_prefix.with_suffix(".warnings.qc.log"), "r") as f:
        assert "BRD3" in f.read() # gene with low correlation

    with h5py.File(output_prefix.with_suffix(".attributions.h5"), "r") as f:
        assert list(f.keys()) == ['attribution', 'gene_mask_end', 'gene_mask_start', 'genes', 'sequence']
        assert f["genes"][:].tolist() == [b"SPI1", b"CD68", b"BRD3"]
        assert f["attribution"].shape == (3, 4, DECIMA_CONTEXT_SIZE)
        assert f["sequence"].shape == (3, DECIMA_CONTEXT_SIZE)
        assert f.attrs["model_name"] == "v1_rep0"
        assert f.attrs["genome"] == "hg38"

    df_peaks = pd.read_csv(output_prefix.with_suffix(".seqlets.bed"), sep="\t", header=None)
    genes = set(df_peaks[3].str.split(".").str[1].str.split("@").str[0])
    assert "SPI1" in genes
    assert "CD68" in genes
    assert "BRD3" in genes

    df_motifs = pd.read_csv(output_prefix.with_suffix(".motifs.tsv"), sep="\t")
    genes = set(df_motifs['peak'].str.split(".").str[1].str.split("@").str[0])
    assert "SPI1" in genes
    assert "CD68" in genes
    assert "BRD3" in genes

@pytest.mark.long_running
def test_cli_attributions_sequences(tmp_path):
    output_prefix = tmp_path / "seqs"
    runner = CliRunner()
    result = runner.invoke(main, [
        "attributions",
        "--seqs", "tests/data/seqs.fasta",
        "--tasks", "cell_type == 'classical monocyte'",
        "--output-prefix", str(output_prefix),
        "--model", "0",
        "--device", device,
    ])
    assert result.exit_code == 0

    assert (output_prefix.with_suffix(".seqlets.bed")).exists()
    assert (output_prefix.with_suffix(".motifs.tsv")).exists()
    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".seqs.fasta")).exists()
    assert (output_prefix.with_suffix(".seqs.fasta.fai")).exists()

    with h5py.File(output_prefix.with_suffix(".attributions.h5"), "r") as f:
        assert list(f.keys()) == ['attribution', 'gene_mask_end', 'gene_mask_start', 'genes', 'sequence']
        assert f["genes"][:].tolist() == [b"CD68", b"SPI1"]
        assert f["attribution"].shape == (2, 4, DECIMA_CONTEXT_SIZE)
        assert f["sequence"].shape == (2, DECIMA_CONTEXT_SIZE)
        assert f.attrs["model_name"] == "v1_rep0"
        assert f.attrs["genome"] == "hg38"

    df_peaks = pd.read_csv(output_prefix.with_suffix(".seqlets.bed"), sep="\t", header=None)
    genes = set(df_peaks[3].str.split(".").str[1].str.split("@").str[0])
    assert "CD68" in genes
    assert "SPI1" in genes

    df_motifs = pd.read_csv(output_prefix.with_suffix(".motifs.tsv"), sep="\t")
    genes = set(df_motifs['peak'].str.split(".").str[1].str.split("@").str[0])
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
        "--model", DEFAULT_ENSEMBLE,
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
        "--model", DEFAULT_ENSEMBLE,
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


@pytest.mark.long_running
def test_cli_modisco_attributions(tmp_path):
    output_prefix = tmp_path / "test_modisco"
    runner = CliRunner()
    result = runner.invoke(main, [
        "modisco-attributions",
        "--output-prefix", str(output_prefix),
        "--top-n-markers", "5",
        "--tasks", "cell_type == 'classical monocyte'",
        "--model", "0",
        "--device", device,
    ])
    assert result.exit_code == 0
    assert (output_prefix.with_suffix(".attributions.h5")).exists()


@pytest.mark.long_running
def test_cli_modisco(tmp_path):
    output_prefix = tmp_path / "test_modisco"
    runner = CliRunner()
    result = runner.invoke(main, [
        "modisco",
        "--output-prefix", str(output_prefix),
        "--tasks", "cell_type == 'classical monocyte'",
        "--top-n-markers", "5",
        "--model", "0",
        "--device", device,
    ])
    assert result.exit_code == 0

    assert (output_prefix.with_suffix(".attributions.h5")).exists()
    assert (output_prefix.with_suffix(".modisco.h5")).exists()
    assert Path(str(output_prefix) + "_report").exists()


@pytest.mark.long_running
def test_cli_vep_attributions(tmp_path):
    output_prefix = tmp_path / "test_vep_attributions"
    runner = CliRunner()
    result = runner.invoke(main, [
        "vep-attribution",
        "-v", "tests/data/variants.tsv",
        "-o", str(output_prefix),
        "--model", "0",
        "--device", device,
    ])
    assert result.exit_code == 0, result.__dict__
    assert Path(str(output_prefix) + ".h5").exists()
