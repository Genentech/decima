import click
from decima import predict_save_attributions


@click.command()
@click.option("-o", "--output_dir", type=str, required=True, help="Directory to save output files")
@click.option("-g", "--genes", type=str, required=False, help="Comma-separated list of gene symbols or IDs to analyze.")
@click.option("--seqs", type=str, required=False, help="Path to a file containing sequences to analyze")
@click.option(
    "--tasks",
    type=str,
    required=False,
    help="Query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte'')",
)
@click.option(
    "--off_tasks", type=str, required=False, help="Optional query string to filter cell types to contrast against."
)
@click.option(
    "--model",
    type=str,
    required=False,
    default=0,
    help="Model to use for attribution analysis either replicate number or path to the model.",
)
@click.option(
    "--metadata",
    type=click.Path(exists=True),
    default=None,
    help="Path to the metadata anndata file. Default: None.",
)
@click.option(
    "--method", type=str, required=False, default="inputxgradient", help="Method to use for attribution analysis."
)
@click.option("--device", type=str, required=False, default=None, help="Device to use for attribution analysis.")
@click.option("--plot_peaks", is_flag=True, default=True, help="Generate peak plots")
@click.option("--plot_seqlogo", is_flag=True, help="Generate sequence logo plots for peaks")
@click.option("--seqlogo_window", type=int, default=50, help="Window size for sequence logo plots")
@click.option("--dpi", type=int, default=100, help="DPI for attribution plots")
def cli_attributions(
    output_dir,
    genes,
    seqs,
    tasks,
    off_tasks,
    model,
    metadata,
    method,
    device,
    plot_peaks,
    plot_seqlogo,
    seqlogo_window,
    dpi,
):
    """
    Generate and save attribution analysis results for a gene or a set of sequences.

    Output files:

        output_dir/

        ├── peaks.bed              # List of attribution peaks in BED format

        ├── peaks_plots/           # Directory containing peak plots

        │   └── {gene}.png         # Plot showing peak locations for each gene

        ├── qc.warnings.log        # QC warnings about prediction reliability

        ├── motifs.tsv             # Detected motifs in peak regions

        ├── attributions.h5        # Raw attribution score matrix

        ├── coverage/              # Directory containing bigwig files

        │   └── {gene}.bw          # Genome browser track of attribution scores

        └── seqlogos/              # Directory containing attribution plots

            └── {peak}.png         # Attribution plot for each peak region

    Examples:

        >>> decima attributions -o output_dir -g SPI1

        >>> decima attributions -o output_dir -g SPI1,CD68 --tasks "cell_type == 'classical monocyte'" --device 0

        >>> decima attributions -o output_dir --seqs tests/data/seqs.fasta --tasks "cell_type == 'classical monocyte'" --device 0

        >>> decima attributions -o output_dir --seqs tests/data/seqs.fasta --tasks "cell_type == 'classical monocyte'" --device 0 --plot_seqlogo
    """
    if model in ["0", "1", "2", "3"]:  # replicate index
        model = int(model)

    predict_save_attributions(
        output_dir=output_dir,
        genes=genes,
        seqs=seqs,
        tasks=tasks,
        off_tasks=off_tasks,
        model=model,
        metadata_anndata=metadata,
        method=method,
        device=device,
        plot_peaks=plot_peaks,
        plot_seqlogo=plot_seqlogo,
        seqlogo_window=seqlogo_window,
        dpi=dpi,
    )
