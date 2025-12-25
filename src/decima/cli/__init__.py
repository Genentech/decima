import logging
import click

from decima.cli.predict_genes import cli_predict_genes
from decima.cli.download import cli_cache, cli_download_weights, cli_download_metadata, cli_download
from decima.cli.attributions import (
    cli_attributions,
    cli_attributions_plot,
    cli_attributions_predict,
    cli_attributions_recursive_seqlet_calling,
)
from decima.cli.query_cell import cli_query_cell
from decima.cli.vep import cli_predict_variant_effect
from decima.cli.finetune import cli_finetune
from decima.cli.vep import cli_vep_ensemble
from decima.cli.vep_attribution import cli_vep_attribution
from decima.cli.modisco import (
    cli_modisco_attributions,
    cli_modisco_patterns,
    cli_modisco_reports,
    cli_modisco_seqlet_bed,
    cli_modisco,
)


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


main.add_command(cli_predict_genes, name="predict-genes")
main.add_command(cli_cache, name="cache")
main.add_command(cli_download_weights, name="download-weights")
main.add_command(cli_download_metadata, name="download-metadata")
main.add_command(cli_download, name="download")
main.add_command(cli_query_cell, name="query-cell")
main.add_command(cli_attributions, name="attributions")
main.add_command(cli_attributions_predict, name="attributions-predict")
main.add_command(cli_attributions_plot, name="attributions-plot")
main.add_command(cli_attributions_recursive_seqlet_calling, name="attributions-recursive-seqlet-calling")
main.add_command(cli_predict_variant_effect, name="vep")
main.add_command(cli_vep_ensemble, name="vep-ensemble")
main.add_command(cli_vep_attribution, name="vep-attribution")
main.add_command(cli_finetune, name="finetune")
main.add_command(cli_modisco, name="modisco")
main.add_command(cli_modisco_attributions, name="modisco-attributions")
main.add_command(cli_modisco_patterns, name="modisco-patterns")
main.add_command(cli_modisco_reports, name="modisco-reports")
main.add_command(cli_modisco_seqlet_bed, name="modisco-seqlet-bed")


if __name__ == "__main__":
    main()
