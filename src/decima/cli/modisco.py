"""
Modisco CLI.

This module contains the CLI for the modisco module performing motif discovery and analysis.

`decima modisco` is the main command for performing motif discovery and analysis.

It includes subcommands for:
- Computing attributions for a given gene or sequence. `modisco-attributions`
- Discovering motifs on the attributions. `modisco-patterns`
- Reporting the motifs. `modisco-reports`
- Extracting the seqlets from the modisco results. `modisco-seqlet-bed`

Examples:
    >>> decima modisco -o output_prefix --tasks "cell_type == 'classical monocyte'" --genes SPI1,CD68

    >>> decima modisco -o output_prefix --tasks "cell_type == 'B cell'" --device 0 --genome hg38

    >>> decima modisco -o output_prefix --genes SPI1 --method saliency --batch-size 2
"""

import click
from typing import List, Optional, Union
from decima.constants import DEFAULT_ENSEMBLE
from decima.cli.callback import parse_model, parse_genes, parse_attributions, parse_metadata
from decima.interpret.modisco import (
    predict_save_modisco_attributions,
    modisco_patterns,
    modisco_reports,
    modisco_seqlet_bed,
    modisco,
)


@click.command()
@click.option("-o", "--output-prefix", type=str, required=True, help="Prefix path to the output files.")
@click.option(
    "--tasks",
    type=str,
    default=None,
    help="Set of tasks respect to which attributions will be computed. If not provided, all tasks will be computed.",
)
@click.option(
    "--off-tasks",
    type=str,
    default=None,
    help="Set of tasks will be subtracted from the attributions to calculate attribution on `specificity` transform. If not provided, all tasks will be computed.",
)
@click.option(
    "--model",
    type=str,
    default=DEFAULT_ENSEMBLE,
    help=f"Model to use for the prediction. Default: {DEFAULT_ENSEMBLE}.",
    callback=parse_model,
    show_default=True,
)
@click.option(
    "--metadata",
    default=None,
    callback=parse_metadata,
    help=f"Path to the metadata anndata file or name of the model. If not provided, the compabilite metadata for the model will be used. Default: {DEFAULT_ENSEMBLE}.",
)
@click.option(
    "--method",
    type=click.Choice(["saliency", "inputxgradient", "integratedgradients"]),
    default="saliency",
    show_default=True,
    help="Method to use for attribution analysis.",
)
@click.option(
    "--transform",
    type=click.Choice(["specificity", "aggregate"]),
    default="specificity",
    show_default=True,
    help="Transform to use for attribution analysis. Available options: 'specificity', 'aggregate'. Specificity transform is recommended for MoDISco to highlight cell-type-specific patterns.",
)
@click.option("--batch-size", type=int, default=1, show_default=True, help="Batch size for the prediction.")
@click.option(
    "--genes",
    type=str,
    default=None,
    callback=parse_genes,
    help="Genes to predict. If not provided, all genes will be predicted.",
)
@click.option(
    "--top-n-markers",
    type=int,
    default=None,
    help="Top n markers to predict. If not provided, all markers will be predicted.",
)
@click.option("--disable-bigwig", is_flag=True, help="Whether to disable bigwig file.")
@click.option(
    "--disable-correct-grad-bigwig", is_flag=True, help="Whether to disable correct gradient for bigwig file."
)
@click.option("--device", type=str, default=None, help="Device to use. If not provided, the best device will be used.")
@click.option(
    "--genome", type=str, default="hg38", show_default=True, help="Genome name or path to the genome fasta file."
)
@click.option("--num-workers", type=int, default=4, show_default=True, help="Number of workers for the prediction.")
def cli_modisco_attributions(
    output_prefix: str,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[Union[str, int]] = DEFAULT_ENSEMBLE,
    metadata: Optional[str] = None,
    method: str = "saliency",
    transform: str = "specificity",
    batch_size: int = 1,
    genes: Optional[str] = None,
    top_n_markers: Optional[int] = None,
    disable_bigwig: bool = False,
    disable_correct_grad_bigwig: bool = False,
    device: Optional[str] = None,
    num_workers: int = 4,
    genome: str = "hg38",
):
    predict_save_modisco_attributions(
        output_prefix=output_prefix,
        tasks=tasks,
        off_tasks=off_tasks,
        model=model,
        metadata_anndata=metadata,
        method=method,
        transform=transform,
        batch_size=batch_size,
        genes=genes,
        top_n_markers=top_n_markers,
        bigwig=not disable_bigwig,
        correct_grad_bigwig=not disable_correct_grad_bigwig,
        num_workers=num_workers,
        device=device,
        genome=genome,
    )


@click.command()
@click.option("-o", "--output-prefix", type=str, required=True, help="Prefix path to the output files.")
@click.option(
    "--attributions",
    type=str,
    required=True,
    callback=parse_attributions,
    help="Comma-separated list of paths to the attributions HDF5 files."
    " If multiple files are provided, they will be averaged."
    " All files must be h5 files generated with `decima modisco-attributions` command.",
)
@click.option(
    "--tasks",
    type=str,
    default=None,
    help="Set of tasks respect to which attributions will be computed. If not provided, all tasks will be computed.",
)
@click.option(
    "--off-tasks",
    type=str,
    default=None,
    help="Set of tasks will be subtracted from the attributions to calculate attribution on `specificity` transform. If not provided, all tasks will be computed.",
)
@click.option("--tss-distance", type=int, default=10_000, show_default=True, help="TSS distance for the prediction.")
@click.option(
    "--metadata",
    default=None,
    callback=parse_metadata,
    help=f"Path to the metadata anndata file or name of the model. Default: {DEFAULT_ENSEMBLE}.",
)
@click.option(
    "--genes",
    type=str,
    default=None,
    callback=parse_genes,
    help="Genes to predict. If not provided, all genes will be predicted.",
)
@click.option(
    "--top-n-markers",
    type=int,
    default=None,
    help="Top n markers to predict. If not provided, all markers will be predicted.",
)
@click.option("--correct-grad", type=bool, default=True, show_default=True, help="Whether to correct gradient.")
@click.option("--num-workers", type=int, default=4, show_default=True, help="Number of workers for the prediction.")
@click.option(
    "--max-seqlets", type=int, default=20_000, show_default=True, help="The maximum number of seqlets per metacluster."
)
@click.option("--n-leiden", type=int, default=16, show_default=True, help="Number of Leiden runs for clustering.")
@click.option(
    "--sliding-window-size", type=int, default=20, show_default=True, help="Seqlet core size (sliding window size)."
)
@click.option(
    "--trim-size", type=int, default=30, show_default=True, help="Length to trim patterns to (trim_to_window_size)."
)
@click.option(
    "--flank-size", type=int, default=5, show_default=True, help="Flank length added to each extracted seqlet."
)
@click.option(
    "--initial-flank-to-add", type=int, default=10, show_default=True, help="Extra flank added when polishing patterns."
)
@click.option(
    "--final-flank-to-add",
    type=int,
    default=0,
    show_default=True,
    help="Additional flank added at the end of motif discovery.",
)
@click.option("--stranded", is_flag=True, help="Treat input as stranded so do not add reverse-complement.")
@click.option(
    "--pattern-type",
    type=click.Choice(["both", "pos", "neg"]),
    default="both",
    show_default=True,
    help="Which pattern signs to compute: both, pos, or neg.",
)
def cli_modisco_patterns(
    output_prefix: str,
    attributions: str,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    tss_distance: int = 1000,
    metadata: Optional[str] = None,
    genes: Optional[List[str]] = None,
    top_n_markers: Optional[int] = None,
    correct_grad: bool = True,
    num_workers: int = 4,
    # modisco parameters
    max_seqlets: int = 20_000,
    n_leiden: int = 16,
    sliding_window_size: int = 20,
    trim_size: int = 30,
    flank_size: int = 5,
    initial_flank_to_add: int = 10,
    final_flank_to_add: int = 0,
    stranded: bool = False,
    pattern_type: str = "both",
):
    modisco_patterns(
        output_prefix=output_prefix,
        attributions=attributions,
        tasks=tasks,
        off_tasks=off_tasks,
        tss_distance=tss_distance,
        metadata_anndata=metadata,
        genes=genes,
        top_n_markers=top_n_markers,
        correct_grad=correct_grad,
        num_workers=num_workers,
        # modisco parameters
        max_seqlets_per_metacluster=max_seqlets,
        n_leiden_runs=n_leiden,
        sliding_window_size=sliding_window_size,
        trim_to_window_size=trim_size,
        flank_size=flank_size,
        initial_flank_to_add=initial_flank_to_add,
        final_flank_to_add=final_flank_to_add,
        stranded=stranded,
        pattern_type=pattern_type,
    )


@click.command()
@click.option("-o", "--output-prefix", type=str, required=True, help="Prefix path to the output files.")
@click.option("--modisco-h5", type=click.Path(exists=True), required=True, help="Path to the modisco HDF5 file.")
@click.option(
    "--meme-motif-db", type=str, default="hocomoco_v13", show_default=True, help="Path to the MEME motif database."
)
@click.option("--img-path-suffix", type=str, default="", help="Suffix path to the output images.")
@click.option("--is-writing-tomtom-matrix", type=bool, default=False, help="Whether to write the Tomtom matrix.")
@click.option("--top-n-matches", type=int, default=3, show_default=True, help="Top n matches to report.")
@click.option("--trim-threshold", type=float, default=0.3, show_default=True, help="Trim threshold.")
@click.option("--trim-min-length", type=int, default=3, show_default=True, help="Trim minimum length.")
@click.option("--tomtomlite", type=bool, default=False, show_default=True, help="Whether to use TomtomLite.")
@click.option("--num-workers", type=int, default=4, show_default=True, help="Number of workers for the prediction.")
def cli_modisco_reports(
    output_prefix: str,
    modisco_h5: str,
    meme_motif_db: str,
    img_path_suffix: str,
    is_writing_tomtom_matrix: bool,
    top_n_matches: int,
    trim_threshold: float,
    trim_min_length: int,
    tomtomlite: bool,
    num_workers: int,
):
    modisco_reports(
        output_prefix=output_prefix,
        modisco_h5=modisco_h5,
        meme_motif_db=meme_motif_db,
        img_path_suffix=img_path_suffix,
        is_writing_tomtom_matrix=is_writing_tomtom_matrix,
        top_n_matches=top_n_matches,
        trim_threshold=trim_threshold,
        trim_min_length=trim_min_length,
        tomtomlite=tomtomlite,
        num_workers=num_workers,
    )


@click.command()
@click.option("-o", "--output-prefix", type=str, required=True, help="Prefix path to the output files.")
@click.option("--modisco-h5", type=click.Path(exists=True), required=True, help="Path to the modisco HDF5 file.")
@click.option(
    "--metadata",
    default=None,
    callback=parse_metadata,
    help=f"Path to the metadata anndata file or name of the model. Default: {DEFAULT_ENSEMBLE}.",
)
@click.option("--trim-threshold", type=float, default=0.2, show_default=True, help="Trim threshold.")
def cli_modisco_seqlet_bed(
    output_prefix: str,
    modisco_h5: str,
    metadata: Optional[str] = None,
    trim_threshold: float = 0.2,
):
    modisco_seqlet_bed(
        output_prefix=output_prefix,
        modisco_h5=modisco_h5,
        metadata_anndata=metadata,
        trim_threshold=trim_threshold,
    )


@click.command()
@click.option("-o", "--output-prefix", type=str, required=True, help="Prefix path to the output files.")
@click.option(
    "--tasks",
    type=str,
    default=None,
    help="Set of tasks respect to which attributions will be computed. If not provided, all tasks will be computed.",
)
@click.option(
    "--off-tasks",
    type=str,
    default=None,
    help="Set of tasks will be subtracted from the attributions to calculate attribution on `specificity` transform. If not provided, all tasks will be computed.",
)
@click.option("--tss-distance", type=int, default=10_000, show_default=True, help="TSS distance for the prediction.")
@click.option(
    "--model",
    type=str,
    default=DEFAULT_ENSEMBLE,
    show_default=True,
    help="`0`, `1`, `2`, `3`, `ensemble` or a path or a comma-separated list of paths to safetensor files. Default: `ensemble`.",
    callback=parse_model,
)
@click.option(
    "--metadata",
    default=None,
    callback=parse_metadata,
    help=f"Path to the metadata anndata file or name of the model. If not provided, the compabilite metadata for the model will be used. Default: {DEFAULT_ENSEMBLE}.",
)
@click.option(
    "--method",
    type=click.Choice(["saliency", "inputxgradient", "integratedgradients"]),
    default="saliency",
    show_default=True,
    help="Method to use for attribution analysis.",
)
@click.option(
    "--transform",
    type=click.Choice(["specificity", "aggregate"]),
    default="specificity",
    show_default=True,
    help="Transform to use for attribution analysis. Available options: 'specificity', 'aggregate'. Specificity transform is recommended for MoDISco to highlight cell-type-specific patterns.",
)
@click.option("--batch-size", type=int, default=1, show_default=True, help="Batch size for the prediction.")
@click.option(
    "--genes",
    type=str,
    show_default=True,
    default=None,
    callback=parse_genes,
    help="Genes to predict. If not provided, all genes will be predicted.",
)
@click.option(
    "--top-n-markers",
    type=int,
    show_default=True,
    default=None,
    help="Top n markers to predict. If not provided, all markers will be predicted.",
)
@click.option("--correct-grad", type=bool, default=True, show_default=True, help="Whether to correct gradient.")
@click.option(
    "--device",
    type=str,
    show_default=True,
    default=None,
    help="Device to use. If not provided, the best device will be used.",
)
@click.option(
    "--genome", type=str, show_default=True, default="hg38", help="Genome name or path to the genome fasta file."
)
@click.option("--num-workers", type=int, show_default=True, default=4, help="Number of workers for the prediction.")
@click.option(
    "--max-seqlets", type=int, default=20_000, show_default=True, help="The maximum number of seqlets per metacluster."
)
@click.option("--n-leiden", type=int, default=16, show_default=True, help="Number of Leiden runs for clustering.")
@click.option(
    "--sliding-window-size", type=int, default=20, show_default=True, help="Seqlet core size (sliding window size)."
)
@click.option(
    "--trim-size", type=int, default=30, show_default=True, help="Length to trim patterns to (trim_to_window_size)."
)
@click.option(
    "--flank-size", type=int, default=5, show_default=True, help="Flank length added to each extracted seqlet."
)
@click.option(
    "--initial-flank-to-add", type=int, default=10, show_default=True, help="Extra flank added when polishing patterns."
)
@click.option(
    "--final-flank-to-add",
    type=int,
    default=0,
    show_default=True,
    help="Additional flank added at the end of motif discovery.",
)
@click.option("--stranded", is_flag=True, help="Treat input as stranded so do not add reverse-complement.")
@click.option(
    "--pattern-type",
    type=click.Choice(["both", "pos", "neg"]),
    default="both",
    show_default=True,
    help="Which pattern signs to compute: both, pos, or neg.",
)
@click.option(
    "--meme-motif-db", type=str, default="hocomoco_v13", show_default=True, help="Path to the MEME motif database."
)
@click.option("--img-path-suffix", type=str, default="", help="Suffix path to the output images.")
@click.option("--is-writing-tomtom-matrix", type=bool, default=False, help="Whether to write the Tomtom matrix.")
@click.option("--top-n-matches", type=int, default=3, show_default=True, help="Top n matches to report.")
@click.option("--trim-threshold", type=float, default=0.3, show_default=True, help="Trim threshold.")
@click.option("--trim-min-length", type=int, default=3, show_default=True, help="Trim minimum length.")
@click.option("--tomtomlite", type=bool, default=False, show_default=True, help="Whether to use TomtomLite.")
@click.option(
    "--seqlet-motif-trim-threshold",
    type=float,
    default=0.2,
    show_default=True,
    help="Trim threshold for motifs in seqlets bed file.",
)
def cli_modisco(
    output_prefix: str,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    tss_distance: int = 10_000,
    model: Optional[Union[str, int]] = DEFAULT_ENSEMBLE,
    metadata: Optional[str] = None,
    method: str = "saliency",
    transform: str = "specificity",
    batch_size: int = 1,
    genes: Optional[str] = None,
    top_n_markers: Optional[int] = None,
    correct_grad: bool = True,
    device: Optional[str] = None,
    num_workers: int = 4,
    genome: str = "hg38",
    # modisco parameters
    max_seqlets: int = 20_000,
    n_leiden: int = 16,
    sliding_window_size: int = 20,
    trim_size: int = 30,
    flank_size: int = 5,
    initial_flank_to_add: int = 10,
    final_flank_to_add: int = 0,
    stranded: bool = False,
    pattern_type: str = "both",
    # reports parameters
    meme_motif_db: str = "hocomoco_v13",
    img_path_suffix: str = "",
    is_writing_tomtom_matrix: bool = False,
    top_n_matches: int = 3,
    trim_threshold: float = 0.3,
    trim_min_length: int = 3,
    tomtomlite: bool = False,
    # seqlet thresholds
    seqlet_motif_trim_threshold: float = 0.2,
):
    modisco(
        output_prefix=output_prefix,
        tasks=tasks,
        off_tasks=off_tasks,
        tss_distance=tss_distance,
        model=model,
        metadata_anndata=metadata,
        method=method,
        transform=transform,
        batch_size=batch_size,
        genes=genes,
        top_n_markers=top_n_markers,
        correct_grad=correct_grad,
        device=device,
        num_workers=num_workers,
        genome=genome,
        # modisco parameters
        max_seqlets_per_metacluster=max_seqlets,
        n_leiden_runs=n_leiden,
        sliding_window_size=sliding_window_size,
        trim_to_window_size=trim_size,
        flank_size=flank_size,
        initial_flank_to_add=initial_flank_to_add,
        final_flank_to_add=final_flank_to_add,
        stranded=stranded,
        pattern_type=pattern_type,
        # reports parameters
        img_path_suffix=img_path_suffix,
        meme_motif_db=meme_motif_db,
        is_writing_tomtom_matrix=is_writing_tomtom_matrix,
        top_n_matches=top_n_matches,
        trim_threshold=trim_threshold,
        trim_min_length=trim_min_length,
        tomtomlite=tomtomlite,
        # seqlet thresholds
        seqlet_motif_trim_threshold=seqlet_motif_trim_threshold,
    )
