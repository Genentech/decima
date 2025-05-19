from click.testing import CliRunner
from decima.cli import main

from conftest import device


def test_cli_main():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0


def test_cli_download():
    runner = CliRunner()
    result = runner.invoke(main, ["download"])
    assert result.exit_code == 0


def test_cli_attributions(tmp_path):
    output_dir = tmp_path / "SPI1"
    runner = CliRunner()
    result = runner.invoke(main, [
        "attributions",
        "--gene", "SPI1",
        "--tasks", "cell_type == 'classical monocyte'",
        "--output_dir", str(output_dir),
        "--model", "0",
        "--device", device
    ])
    assert result.exit_code == 0

    assert (output_dir / "peaks.bed").exists()
    assert (output_dir / "peaks.png").exists()
    assert (output_dir / "attributions.bigwig").exists()
    assert (output_dir / "attributions.npz").exists()
    assert (output_dir / "attributions_seq_logos").is_dir()
    assert (output_dir / "motifs.tsv").exists()
    assert (output_dir / "qc.log").exists()
