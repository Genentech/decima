"""
Attributions CLI.

This module contains the CLI for the attributions module performing attribution analysis, seqlet calling, and motif discovery.

`decima attributions` is the main command for performing attribution analysis, seqlet calling, and motif discovery.

It includes subcommands for:
- Predicting attributions for a given gene or sequence. `attributions-predict`
- Seqlet calling on the attributions. `attributions`
- Motif discovery on the attributions. `attributions-recursive-seqlet-calling`
- Plotting the attributions. `attributions-plot`

Examples:
    >>> decima attributions -o output_prefix -t tasks -o off_tasks -m model -m metadata -m method -m transform -m batch_size -m genes -m top_n_markers -m disable_bigwig -m disable_correct_grad_bigwig -m device -m genome -m num_workers
    ...
"""

import click
from decima.constants import DEFAULT_ENSEMBLE
from decima.cli.callback import parse_genes, parse_model, parse_attributions, parse_metadata
from decima.interpret.attributions import (
    plot_attributions,
    predict_save_attributions,
    recursive_seqlet_calling,
    predict_attributions_seqlet_calling,
)


@click.command()
@click.option("-o", "--output-prefix", type=str, required=True, help="Prefix path to the output files")
@click.option(
    "--tasks",
    type=str,
    required=False,
    default=None,
    help="Query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte''). If not provided, all tasks will be analyzed.",
    show_default=True,
)
@click.option(
    "--off-tasks",
    type=str,
    required=False,
    default=None,
    help="Optional query string to filter cell types to contrast against. If not provided, no contrast will be performed.",
)
@click.option(
    "--model",
    type=str,
    required=False,
    default=DEFAULT_ENSEMBLE,
    callback=parse_model,
    help="Model to use for attribution analysis either replicate number or path to the model.",
    show_default=True,
)
@click.option(
    "--metadata",
    default=None,
    callback=parse_metadata,
    help="Path to the metadata anndata file or name of the model. If not provided, the compabilite metadata for the model will be used.",
    show_default=True,
)
@click.option(
    "--method",
    type=str,
    required=False,
    default="inputxgradient",
    help="Method to use for attribution analysis.",
    show_default=True,
)
@click.option(
    "--transform",
    type=click.Choice(["specificity", "aggregate"]),
    required=False,
    default="specificity",
    help="Transform to use for attribution analysis.",
    show_default=True,
)
@click.option(
    "--batch-size",
    type=int,
    required=False,
    default=1,
    help="Batch size for attribution analysis.",
    show_default=True,
)
@click.option(
    "-g",
    "--genes",
    type=str,
    required=False,
    help="Comma-separated list of gene symbols or IDs to analyze.",
    callback=parse_genes,
    show_default=True,
)
@click.option(
    "--seqs",
    type=str,
    required=False,
    help="Path to a fasta file containing sequences to analyze.",
)
@click.option(
    "--top-n-markers",
    type=int,
    default=None,
    help="Top n markers to predict. If not provided, all markers will be predicted.",
    show_default=True,
)
@click.option(
    "--num-workers",
    type=int,
    required=False,
    default=4,
    help="Number of workers for attribution analysis.",
    show_default=True,
)
@click.option(
    "--device",
    type=str,
    required=False,
    default=None,
    help="Device to use for attribution analysis. If not provided, `cuda` will be used if available, otherwise `cpu` will be used.",
    show_default=True,
)
@click.option(
    "--genome",
    type=str,
    default="hg38",
    help="Genome name or path to the genome fasta file.",
    show_default=True,
)
def cli_attributions_predict(
    output_prefix,
    tasks,
    off_tasks,
    model,
    metadata,
    method,
    transform,
    batch_size,
    genes,
    seqs,
    top_n_markers,
    num_workers,
    device,
    genome,
):
    """Predict and save attributions for specified genes or sequences using the chosen decima model.

    Output files:

    ├── {output_prefix}.attributions.h5      # Raw attribution score matrix per gene.

    └── {output_prefix}.attributions.bigwig  # Genome browser track of attribution as bigwig file obtained with averaging the attribution scores across the genes for genomics coordinates.
    """
    predict_save_attributions(
        output_prefix=output_prefix,
        tasks=tasks,
        off_tasks=off_tasks,
        model=model,
        metadata_anndata=metadata,
        method=method,
        transform=transform,
        batch_size=batch_size,
        genes=genes,
        seqs=seqs,
        top_n_markers=top_n_markers,
        bigwig=True,
        correct_grad_bigwig=False,
        num_workers=num_workers,
        device=device,
        genome=genome,
    )


@click.command()
@click.option("-o", "--output-prefix", type=str, required=True, help="Prefix path to the output files")
@click.option(
    "-g",
    "--genes",
    type=str,
    required=False,
    callback=parse_genes,
    help="Comma-separated list of gene symbols or IDs to analyze.",
)
@click.option("--seqs", type=str, required=False, help="Path to a file containing sequences to analyze")
@click.option(
    "--tasks",
    type=str,
    required=False,
    help="Query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte'')",
)
@click.option(
    "--off-tasks", type=str, required=False, help="Optional query string to filter cell types to contrast against."
)
@click.option(
    "--model",
    type=str,
    required=False,
    default=DEFAULT_ENSEMBLE,
    callback=parse_model,
    help="Model to use for attribution analysis either replicate number or path to the model.",
    show_default=True,
)
@click.option(
    "--metadata",
    callback=parse_metadata,
    default=None,
    help="Path to the metadata anndata file or name of the model. If not provided, the compabilite metadata for the model will be used.",
)
@click.option(
    "--method", type=str, required=False, default="inputxgradient", help="Method to use for attribution analysis."
)
@click.option(
    "--transform",
    type=click.Choice(["specificity", "aggregate"]),
    required=False,
    default="specificity",
    help="Transform to use for attribution analysis.",
)
@click.option("--num-workers", type=int, required=False, default=4, help="Number of workers for attribution analysis.")
@click.option("--tss-distance", type=int, required=False, default=None, help="TSS distance for attribution analysis.")
@click.option("--batch-size", type=int, required=False, default=1, help="Batch size for attribution analysis.")
@click.option(
    "--top-n-markers",
    type=int,
    default=None,
    help="Top n markers to predict. If not provided, all markers will be predicted.",
)
@click.option("--threshold", type=float, required=False, default=5e-4, help="Threshold for attribution analysis.")
@click.option("--min-seqlet-len", type=int, required=False, default=4, help="Minimum length for seqlet calling.")
@click.option("--max-seqlet-len", type=int, required=False, default=25, help="Maximum length for seqlet calling.")
@click.option("--additional-flanks", type=int, required=False, default=0, help="Additional flanks for seqlet calling.")
@click.option(
    "--pattern-type",
    type=click.Choice(["both", "pos", "neg"]),
    required=False,
    default="both",
    help="Type of pattern to call.",
)
@click.option(
    "--meme-motif-db", type=str, default="hocomoco_v13", show_default=True, help="Path to the MEME motif database."
)
@click.option("--device", type=str, required=False, default=None, help="Device to use for attribution analysis.")
@click.option(
    "--genome", type=str, show_default=True, default="hg38", help="Genome name or path to the genome fasta file."
)
def cli_attributions(
    output_prefix,
    genes,
    seqs,
    tasks,
    off_tasks,
    model,
    metadata,
    method,
    transform,
    num_workers,
    tss_distance,
    batch_size,
    top_n_markers,
    threshold,
    min_seqlet_len,
    max_seqlet_len,
    additional_flanks,
    pattern_type,
    meme_motif_db,
    device,
    genome,
):
    """Generate and save attribution analysis results for a gene or a set of sequences and perform seqlet calling on the attributions.

    Output files:

        ├── {output_prefix}.attributions.h5      # Raw attribution score matrix per gene.

        ├── {output_prefix}.attributions.bigwig  # Genome browser track of attribution as bigwig file.

        ├── {output_prefix}.seqlets.bed          # List of attribution peaks in BED format.

        ├── {output_prefix}.motifs.tsv           # Detected motifs in peak regions.

        └── {output_prefix}.warnings.qc.log      # QC warnings about prediction reliability.

    Examples:

        >>> decima attributions -o output_prefix -g SPI1

        >>> decima attributions -o output_prefix -g SPI1,CD68 --tasks "cell_type == 'classical monocyte'" --device 0

        >>> decima attributions -o output_prefix --seqs tests/data/seqs.fasta --tasks "cell_type == 'classical monocyte'" --device 0
    """
    predict_attributions_seqlet_calling(
        output_prefix=output_prefix,
        genes=genes,
        seqs=seqs,
        tasks=tasks,
        off_tasks=off_tasks,
        model=model,
        metadata_anndata=metadata,
        method=method,
        transform=transform,
        num_workers=num_workers,
        device=device,
        batch_size=batch_size,
        top_n_markers=top_n_markers,
        tss_distance=tss_distance,
        threshold=threshold,
        min_seqlet_len=min_seqlet_len,
        max_seqlet_len=max_seqlet_len,
        additional_flanks=additional_flanks,
        pattern_type=pattern_type,
        meme_motif_db=meme_motif_db,
        genome=genome,
    )


@click.command()
@click.option("-o", "--output-prefix", type=str, required=True, help="Prefix path to the output files")
@click.option(
    "--attributions", type=str, callback=parse_attributions, required=True, help="Path to the attribution files"
)
@click.option(
    "--tasks",
    type=str,
    required=False,
    help="Query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte'')",
)
@click.option(
    "--off-tasks", type=str, required=False, help="Optional query string to filter cell types to contrast against."
)
@click.option("--tss-distance", type=int, required=False, default=None, help="TSS distance for attribution analysis.")
@click.option(
    "--metadata",
    callback=parse_metadata,
    default=None,
    help="Path to the metadata anndata file or name of the model. If not provided, the compabilite metadata for the model will be used.",
)
@click.option(
    "--genes",
    type=str,
    required=False,
    callback=parse_genes,
    help="Comma-separated list of gene symbols or IDs to analyze.",
)
@click.option(
    "--top-n-markers",
    type=int,
    required=False,
    default=None,
    help="Top n markers to predict. If not provided, all markers will be predicted.",
)
@click.option("--num-workers", type=int, required=False, default=4, help="Number of workers for attribution analysis.")
@click.option(
    "--agg-func", type=str, required=False, default="mean", help="Aggregation function for attribution analysis."
)
@click.option("--threshold", type=float, required=False, default=5e-4, help="Threshold for attribution analysis.")
@click.option("--min-seqlet-len", type=int, required=False, default=4, help="Minimum length for seqlet calling.")
@click.option("--max-seqlet-len", type=int, required=False, default=25, help="Maximum length for seqlet calling.")
@click.option("--additional-flanks", type=int, required=False, default=0, help="Additional flanks for seqlet calling.")
@click.option(
    "--pattern-type",
    type=click.Choice(["both", "pos", "neg"]),
    required=False,
    default="both",
    help="Type of pattern to call.",
)
@click.option("--custom-genome", is_flag=True, help="Use custom genome")
@click.option(
    "--meme-motif-db", type=str, default="hocomoco_v13", show_default=True, help="Path to the MEME motif database."
)
def cli_attributions_recursive_seqlet_calling(
    output_prefix,
    attributions,
    tasks,
    off_tasks,
    tss_distance,
    metadata,
    genes,
    top_n_markers,
    num_workers,
    agg_func,
    threshold,
    min_seqlet_len,
    max_seqlet_len,
    additional_flanks,
    pattern_type,
    custom_genome,
    meme_motif_db,
):
    """Performs recursive seqlet calling on the pre computed attributions.

    Output files:

        ├── {output_prefix}.seqlets.bed          # List of attribution peaks in BED format.

        ├── {output_prefix}.motifs.tsv           # Detected motifs in peak regions.

        └── {output_prefix}.warnings.qc.log      # QC warnings about prediction reliability.

    Examples:

        >>> decima attributions-recursive-seqlet-calling --attributions attributions_0.h5,attributions_1.h5 -o output_prefix  --genes SPI1
    """
    recursive_seqlet_calling(
        output_prefix=output_prefix,
        attributions=attributions,
        tasks=tasks,
        off_tasks=off_tasks,
        tss_distance=tss_distance,
        metadata_anndata=metadata,
        genes=genes,
        top_n_markers=top_n_markers,
        num_workers=num_workers,
        agg_func=agg_func,
        threshold=threshold,
        min_seqlet_len=min_seqlet_len,
        max_seqlet_len=max_seqlet_len,
        additional_flanks=additional_flanks,
        pattern_type=pattern_type,
        custom_genome=custom_genome,
        meme_motif_db=meme_motif_db,
    )


@click.command()
@click.option("-o", "--output-prefix", type=str, required=True, help="Prefix path to the output files")
@click.option(
    "-g",
    "--genes",
    type=str,
    required=False,
    callback=parse_genes,
    help="Comma-separated list of gene symbols or IDs to analyze.",
)
@click.option(
    "--metadata",
    callback=parse_metadata,
    default=None,
    help="Path to the metadata anndata file or name of the model. If not provided, the compabilite metadata for the model will be used.",
)
@click.option("--tss-distance", type=int, required=False, default=None, help="TSS distance for attribution analysis.")
@click.option("--seqlogo-window", type=int, default=50, help="Window size for sequence logo plots")
@click.option("--custom-genome", is_flag=True, help="Use custom genome")
@click.option("--dpi", type=int, default=100, help="DPI for attribution plots")
def cli_attributions_plot(
    output_prefix,
    genes,
    metadata,
    tss_distance,
    seqlogo_window,
    custom_genome,
    dpi,
):
    """Plots the attributions for the specified genes.

    Output files:

        ├── {output_prefix}.attributions.png  # Attributions plot.

        └── {output_prefix}.seqlogo.png       # Sequence logo plot.

        Examples:

            >>> decima attributions-plot -o output_prefix -g SPI1
    """
    plot_attributions(
        output_prefix=output_prefix,
        genes=genes,
        metadata_anndata=metadata,
        tss_distance=tss_distance,
        seqlogo_window=seqlogo_window,
        custom_genome=custom_genome,
        dpi=dpi,
    )
