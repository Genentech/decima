import logging
import click

from decima.cli.predict_genes import cli_predict_genes
from decima.cli.download import cli_download
from decima.cli.attributions import cli_attributions
from decima.cli.query_cell import cli_query_cell
from decima.cli.vep import cli_predict_variant_effect
from decima.cli.finetune import cli_finetune
from decima.cli.vep import cli_vep_ensemble


logger = logging.getLogger("decima")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@click.group()
def main():
    """Decima CLI interface."""
    pass


# main.add_command(cli_finetune, name="finetune")
main.add_command(cli_predict_genes, name="predict-genes")
main.add_command(cli_download, name="download")
main.add_command(cli_attributions, name="attributions")
main.add_command(cli_query_cell, name="query-cell")
main.add_command(cli_predict_variant_effect, name="vep")
main.add_command(cli_finetune, name="finetune")
main.add_command(cli_vep_ensemble, name="vep-ensemble")


if __name__ == "__main__":
    main()
