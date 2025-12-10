"""
Interpretation of Variant Effect Prediction with Attribution Analysis CLI.
"""

import click
from decima.constants import DECIMA_CONTEXT_SIZE, DEFAULT_ENSEMBLE, MODEL_METADATA
from decima.cli.callback import parse_metadata, parse_model
from decima.vep.attributions import variant_effect_attribution


@click.command()
@click.option(
    "-v",
    "--variants",
    type=click.Path(exists=True),
    help="Path to the variant file .vcf file. VCF file need to be normalized. Try normalizing th vcf file incase of an error. `bcftools norm -f ref.fasta input.vcf.gz -o output.vcf.gz`",
)
@click.option("-o", "--output_h5", type=click.Path(), help="Path to the output h5 file.")
@click.option("--tasks", type=str, default=None, help="Tasks to predict. If not provided, all tasks will be predicted.")
@click.option(
    "--off-tasks",
    type=str,
    default=None,
    help="Tasks to contrast against. If not provided, no contrast will be performed.",
)
@click.option(
    "--model",
    default=0,
    callback=parse_model,
    help=f"Model to use for attribution analysis. Available options: {list(MODEL_METADATA.keys())}.",
)
@click.option(
    "--metadata",
    default=None,
    callback=parse_metadata,
    help=f"Path to the metadata anndata file or name of the model. If not provided, the compabilite metadata for the model will be used. Default: {DEFAULT_ENSEMBLE}.",
)
@click.option(
    "--method",
    type=str,
    default="inputxgradient",
    help="Method to use for attribution analysis. Available options: 'inputxgradient', 'saliency', 'integratedgradients'.",
)
@click.option(
    "--transform",
    type=click.Choice(["specificity", "aggregate"]),
    default="specificity",
    help="Transform to use for attribution analysis. Available options: 'specificity', 'aggregate'.",
)
@click.option(
    "--device", type=str, default=None, help="Device to use. Default: None which automatically selects the best device."
)
@click.option("--num-workers", type=int, default=4, help="Number of workers for the loader. Default: 4")
@click.option("--distance-type", type=str, default="tss", help="Type of distance. Default: tss.")
@click.option(
    "--min-distance",
    type=float,
    default=0,
    help="Minimum distance from the end of the gene. Default: 0.",
)
@click.option(
    "--max-distance",
    type=float,
    default=DECIMA_CONTEXT_SIZE,
    help=f"Maximum distance from the TSS. Default: {DECIMA_CONTEXT_SIZE}.",
)
@click.option(
    "--gene-col",
    type=str,
    default=None,
    help="Column name for gene names. Default: None.",
)
@click.option("--genome", type=str, default="hg38", help="Genome build. Default: hg38.")
@click.option(
    "--float-precision",
    type=str,
    default="32",
    help="Floating-point precision to be used in calculations. Avaliable options include: '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '32', '16', and 'bf16'.",
)
def cli_vep_attribution(
    variants,
    output_h5,
    tasks,
    off_tasks,
    model,
    metadata,
    method,
    transform,
    device,
    num_workers,
    distance_type,
    min_distance,
    max_distance,
    gene_col,
    genome,
    float_precision,
):
    """Predict variant effect and save to parquet

    Examples:

        >>> decima vep-attribution -v "data/sample.vcf" -o "vep_results.h5"
    """
    variant_effect_attribution(
        variants,
        output_h5=output_h5,
        tasks=tasks,
        off_tasks=off_tasks,
        model=model,
        metadata_anndata=metadata,
        method=method,
        transform=transform,
        num_workers=num_workers,
        device=device,
        distance_type=distance_type,
        min_distance=min_distance,
        max_distance=max_distance,
        gene_col=gene_col,
        genome=genome,
        float_precision=float_precision,
    )
