import click
from decima.tools.inference import predict_gene_expression


@click.command()
@click.option("-o", "--output", type=click.Path(), help="Path to the output h5ad file.")
@click.option(
    "--genes",
    type=str,
    default=None,
    help="List of genes to predict. Default: None (all genes). If provided, only these genes will be predicted.",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default="ensemble",
    help="Path to the model checkpoint: `0`, `1`, `2`, `3`, `ensemble` or `path/to/model.ckpt`.",
)
@click.option(
    "--metadata",
    type=click.Path(exists=True),
    default=None,
    help="Path to the metadata anndata file. Default: None.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use. Default: None which automatically selects the best device.",
)
@click.option("--batch-size", type=int, default=8, help="Batch size for the model. Default: 8")
@click.option("--num-workers", type=int, default=4, help="Number of workers for the loader. Default: 4")
@click.option("--max_seq_shift", default=0, help="Maximum jitter for augmentation.")
@click.option("--genome", type=str, default="hg38", help="Genome build. Default: hg38.")
@click.option(
    "--save-replicates",
    is_flag=True,
    help="Save the replicates in the output parquet file. Default: False.",
)
@click.option(
    "--float-precision",
    type=str,
    default="32",
    help="Floating-point precision to be used in calculations. Avaliable options include: '16-true', '16-mixed', 'bf16-true', 'bf16-mixed', '32-true', '64-true', '32', '16', and 'bf16'.",
)
def cli_predict_genes(
    output,
    genes,
    model,
    metadata,
    device,
    batch_size,
    num_workers,
    max_seq_shift,
    genome,
    save_replicates,
    float_precision,
):
    if model in ["0", "1", "2", "3"]:
        model = int(model)

    if isinstance(device, str) and device.isdigit():
        device = int(device)

    if genes is not None:
        genes = genes.split(",")

    if save_replicates and (model != "ensemble"):
        raise ValueError("`--save-replicates` is only supported for ensemble model (`--model ensemble`).")

    ad = predict_gene_expression(
        genes=genes,
        model=model,
        metadata_anndata=metadata,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        max_seq_shift=max_seq_shift,
        genome=genome,
        save_replicates=save_replicates,
        float_precision=float_precision,
    )
    ad.write_h5ad(output)
