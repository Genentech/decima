import click
from decima import predict_save_attributions


@click.command()
@click.option("-o", "--output_dir", type=str, required=True, help="Directory to save output files")
@click.option("-g", "--gene", type=str, required=True, help="Gene symbol or ID to analyze")
@click.option(
    "--tasks",
    type=str,
    required=True,
    help="Query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte'')",
)
@click.option(
    "--off_tasks", type=str, required=False, help="Optional query string to filter cell types to contrast against."
)
@click.option(
    "--model",
    type=int,
    required=False,
    default=0,
    help="Model to use for attribution analysis either replicate number of path to the model.",
)
@click.option("--device", type=str, required=False, default="cpu", help="Device to use for attribution analysis.")
def attributions(output_dir, gene, tasks, off_tasks, model, device):
    """
    Generate and save attribution analysis results for a gene.

    Examples:

        >>> decima attributions -o output_dir -g SPI1 --tasks "cell_type == 'classical monocyte'" --device 0

    Output files:

        output_dir/

        ├── peaks.bed                # List of attribution peaks in BED format

        ├── peaks.png                # Plot showing peak locations

        ├── qc.log                   # QC warnings about prediction reliability

        ├── motifs.tsv               # Detected motifs in peak regions

        ├── attributions.npz         # Raw attribution score matrix

        ├── attributions.bigwig      # Genome browser track of attribution scores

        └── attributions_seq_logos/  # Directory containing attribution plots

            └── {peak}.png           # Attribution plot for each peak region
    """
    predict_save_attributions(output_dir, gene, tasks, off_tasks, model, device)
