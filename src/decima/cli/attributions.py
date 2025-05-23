import click
from decima import predict_save_attributions


@click.command()
@click.option("-o", "--output_dir", type=str, required=True, help="Directory to save output files")
@click.option("-g", "--gene", type=str, required=False, help="Gene symbol or ID to analyze")
@click.option("--seqs", type=str, required=False, help="Path to a file containing sequences to analyze")
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
@click.option("--plot_seqlogo", is_flag=True, help="Generate sequence logo plots for peaks")
@click.option("--seqlogo_window", type=int, default=50, help="Window size for sequence logo plots")
@click.option("--dpi", type=int, default=100, help="DPI for attribution plots")
def attributions(output_dir, gene, seqs, tasks, off_tasks, model, device, plot_seqlogo, seqlogo_window, dpi):
    """
    Generate and save attribution analysis results for a gene or a set of sequences.

    Examples:
        >>> decima attributions -o output_dir -g SPI1 --tasks "cell_type == 'classical monocyte'" --device 0

    Output files:

        output_dir/

        ├── peaks.bed                # List of attribution peaks in BED format

        ├── peaks.png                # Plot showing peak locations

        ├── qc.log                   # QC warnings about prediction reliability

        ├── motifs.tsv               # Detected motifs in peak regions

        ├── attributions.h5          # Raw attribution score matrix

        ├── attributions.bigwig      # Genome browser track of attribution scores

        └── seqlogos/               # Directory containing attribution plots

            └── {peak}.png           # Attribution plot for each peak region
    """
    predict_save_attributions(
        output_dir=output_dir,
        genes=gene,
        seqs=seqs,
        tasks=tasks,
        off_tasks=off_tasks,
        model=model,
        device=device,
        plot_seqlogo=plot_seqlogo,
        seqlogo_window=seqlogo_window,
        dpi=dpi,
    )
