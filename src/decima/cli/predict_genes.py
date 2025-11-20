"""
Predict Genes CLI.

This module contains the CLI for predicting the gene expression of a given gene or sequence.

`decima predict-genes` is the command for predicting the gene expression of a given gene or sequence.
"""

import click
from pathlib import Path
from decima.constants import DEFAULT_ENSEMBLE
from decima.cli.callback import parse_model, parse_genes, validate_save_replicates, parse_metadata
from decima.tools.inference import predict_gene_expression


@click.command()
@click.option("-o", "--output", type=click.Path(), help="Path to the output h5ad file.")
@click.option(
    "--genes",
    type=str,
    default=None,
    callback=parse_genes,
    help="Comma-separated list of genes to predict. Default: None (all genes). If provided, only these genes will be predicted.",
)
@click.option(
    "-m",
    "--model",
    type=str,
    default=DEFAULT_ENSEMBLE,
    show_default=True,
    callback=parse_model,
    help=f"`0`, `1`, `2`, `3`, `{DEFAULT_ENSEMBLE}` or a path or a comma-separated list of paths to checkpoint files",
)
@click.option(
    "--metadata",
    default=None,
    callback=parse_metadata,
    help="Path to the metadata anndata file or name of the model. If not provided, the compabilite metadata for the model will be used. Default: {DEFAULT_ENSEMBLE}.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to use. Default: None which automatically selects the best device.",
)
@click.option("--batch-size", type=int, default=1, help="Batch size for the model. Default: 8")
@click.option("--num-workers", type=int, default=4, help="Number of workers for the loader. Default: 4")
@click.option("--max_seq_shift", default=0, help="Maximum jitter for augmentation.")
@click.option("--genome", type=str, default="hg38", help="Genome build. Default: hg38.")
@click.option(
    "--save-replicates",
    is_flag=True,
    callback=validate_save_replicates,
    help="Save the replicates in the output h5ad file. Default: False. Only supported for ensemble models.",
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
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ad.write_h5ad(output)
