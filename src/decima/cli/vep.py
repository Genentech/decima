import click
from decima.constants import DECIMA_CONTEXT_SIZE
from decima.utils.dataframe import ensemble_predictions
from decima.vep import predict_variant_effect


@click.command()
@click.option(
    "-v",
    "--variants",
    type=click.Path(exists=True),
    help="Path to the variant file .vcf file. VCF file need to be normalized. Try normalizing th vcf file incase of an error. `bcftools norm -f ref.fasta input.vcf.gz -o output.vcf.gz`",
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
    type=str,
    default="ensemble",
    help="Model to use for variant effect prediction either replicate number or path to the model.",
)
@click.option(
    "--metadata",
    type=click.Path(exists=True),
    default=None,
    help="Path to the metadata anndata file. Default: None.",
)
@click.option(
    "--device", type=str, default=None, help="Device to use. Default: None which automatically selects the best device."
)
@click.option("--batch-size", type=int, default=8, help="Batch size for the model. Default: 8")
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
@click.option(
    "--save-replicates",
    is_flag=True,
    help="Save the replicates in the output parquet file. Default: False.",
)
@click.option(
    "--disable-reference-cache",
    is_flag=True,
    help="Disables the reference cache which significantly speeds up the computation by caching the reference expression predictios in the metadata.",
)
@click.option(
    "--float-precision",
    type=str,
    default="32",
    help="Floating-point precision to be used in calculations. Avaliable options include: '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '32', '16', and 'bf16'.",
)
def cli_predict_variant_effect(
    variants,
    output_pq,
    tasks,
    chunksize,
    model,
    metadata,
    device,
    batch_size,
    num_workers,
    distance_type,
    min_distance,
    max_distance,
    include_cols,
    gene_col,
    genome,
    save_replicates,
    disable_reference_cache,
    float_precision,
):
    """Predict variant effect and save to parquet

    Examples:

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet"

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --tasks "cell_type == 'classical monocyte'" # only predict for classical monocytes

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --device 0 # use device gpu device 0

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --include-cols "gene_name,gene_id" # include gene_name and gene_id columns in the output

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --gene-col "gene_name" # use gene_name column as gene names if these option passed genes and variants mapped based on these column not based on the genomic locus based on the annotaiton.

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --distance-type tss --min-distance 50000 --max-distance 100000 # predict for variants within 50kb of the TSS and 100kb of the TSS

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --save-replicates # save the replicates in the output parquet file

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --genome "hg38" # use hg38 genome build

        >>> decima vep -v "data/sample.vcf" -o "vep_results.parquet" --genome "path/to/fasta/hg38.fa"  # use custom genome build
    """
    reference_cache = not disable_reference_cache

    if model in ["0", "1", "2", "3"]:  # replicate index
        model = int(model)

    if isinstance(device, str) and device.isdigit():
        device = int(device)

    if include_cols:
        include_cols = include_cols.split(",")

    if save_replicates and (model != "ensemble"):
        raise ValueError("`--save-replicates` is only supported for ensemble model (`--model ensemble`).")

    predict_variant_effect(
        variants,
        output_pq=output_pq,
        tasks=tasks,
        model=model,
        metadata_anndata=metadata,
        chunksize=chunksize,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        distance_type=distance_type,
        min_distance=min_distance,
        max_distance=max_distance,
        include_cols=include_cols,
        gene_col=gene_col,
        genome=genome,
        save_replicates=save_replicates,
        reference_cache=reference_cache,
        float_precision=float_precision,
    )


@click.command()
@click.option("-f", "--files", type=str, help="Path to the parquet files to ensemble. Can be passed multiple times.")
@click.option("-o", "--output_pq", type=click.Path(), help="Path to the output parquet file.")
@click.option(
    "--save-replicates",
    default=False,
    type=bool,
    is_flag=True,
    help="Save the replicates in the output parquet file. Default: False.",
)
def cli_vep_ensemble(files, output_pq, save_replicates=False):
    """Ensemble variant effect predictions from multiple parquet files

    Examples:

        >>> decima vep-ensemble -f "data/sample_rep0.parquet,data/sample_rep1.parquet,data/sample_rep2.parquet" -o "vep_results.parquet"

        >>> decima vep-ensemble -f "data/sample_rep*.parquet" -o "vep_results.parquet" --save-replicates
    """
    if "," in files:
        files = files.split(",")
    ensemble_predictions(files=files, output_pq=output_pq, save_replicates=save_replicates)
