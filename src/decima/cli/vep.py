import click
from decima.constants import DECIMA_CONTEXT_SIZE
from decima.vep import predict_variant_effect


@click.command()
@click.option(
    "-v",
    "--variants",
    type=click.Path(exists=True),
    help="Path to the variant file .vcf file",
)
@click.option("-o", "--output_pq", type=click.Path(), help="Path to the output parquet file.")
@click.option("--tasks", type=str, default=None, help="Tasks to predict. If not provided, all tasks will be predicted.")
@click.option(
    "--chunksize",
    type=int,
    default=10_000,
    help="Number of variants to process in each chunk. Loading variants in chunks is more memory efficient."
    "This chuck of variants will be process and saved to output parquet file before contineus to next chunk. Default: 10_000.",
)
@click.option(
    "--model",
    type=int,
    default=0,
    help="Model to use for variant effect prediction either replicate number or path to the model.",
)
@click.option(
    "--device", type=str, default=None, help="Device to use. Default: None which automatically selects the best device."
)
@click.option("--batch-size", type=int, default=1, help="Batch size for the model. Default: 1.")
@click.option("--num-workers", type=int, default=1, help="Number of workers for the loader. Default: 1.")
@click.option(
    "--max-distance",
    type=float,
    default=DECIMA_CONTEXT_SIZE,
    help=f"Maximum distance from the TSS. Default: {DECIMA_CONTEXT_SIZE}.",
)
@click.option("--max-distance-type", type=str, default="tss", help="Type of maximum distance. Default: tss.")
@click.option(
    "--include-cols",
    type=str,
    default=None,
    help="Columns to include in the output in the original tsv file to include in the output parquet file. Default: None.",
)
@click.option(
    "--gene-col",
    type=str,
    default=None,
    help="Column name for gene names. Default: None.",
)
@click.option("--genome", type=str, default="hg38", help="Genome build. Default: hg38.")
def cli_predict_variant_effect(
    variants,
    output_pq,
    tasks,
    chunksize,
    model,
    device,
    batch_size,
    num_workers,
    max_distance,
    max_distance_type,
    # max_dist_genebody, # TODO: only one should be provided
    include_cols,
    gene_col,
    genome,
):
    """Predict variant effect and save to parquet

    Examples:

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet"

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --tasks "cell_type == 'classical monocyte'" # only predict for classical monocytes

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --device 0 # use device gpu device 0

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --include-cols "gene_name,gene_id" # include gene_name and gene_id columns in the output

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --gene-col "gene_name" # use gene_name column as gene names if these option passed genes and variants mapped based on these column not based on the genomic locus based on the annotaiton.
    """
    if model in ["0", "1", "2", "3"]:  # replicate index
        model = int(model)

    predict_variant_effect(
        variants,
        output_pq=output_pq,
        tasks=tasks,
        model=model,
        chunksize=chunksize,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_distance=max_distance,
        max_distance_type=max_distance_type,
        include_cols=include_cols,
        gene_col=gene_col,
        genome=genome,
    )
