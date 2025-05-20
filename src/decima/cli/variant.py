import pandas as pd
import click
from decima.vep import predict_variant_effect


@click.command()
@click.argument("variant_file", type=click.Path(exists=True))
@click.argument("output_pq", type=click.Path())
@click.option("--tasks", type=str, default=None)
@click.option("--chuck_size", type=int, default=10_000)
@click.option("--model", type=int, default=0)
@click.option("--device", type=str, default="cpu")
@click.option("--batch-size", type=int, default=8)
@click.option("--num-workers", type=int, default=16)
@click.option("--min-from-end", type=int, default=0)
@click.option("--max-dist-tss", type=int, default=float("inf"))
@click.option("--include-cols", type=str, default=None)
def cli_predict_variant_effect(
    variant_file, output_pq, tasks, model, device, batch_size, num_workers, min_from_end, max_dist_tss, include_cols
):
    if variant_file.endswith(".tsv"):
        df_variant = pd.read_csv(variant_file, sep="\t")
        assert (
            "chrom" in df_variant.columns
            and "pos" in df_variant.columns
            and "ref" in df_variant.columns
            and "alt" in df_variant.columns
        ), "TSV file must have chrom, pos, ref, alt columns"
    elif variant_file.endswith(".vcf"):
        raise NotImplementedError("VCF input not implemented yet")
    else:
        raise ValueError(
            f"Unsupported file extension: {variant_file}. Must be .tsv with columns: chrom, pos, ref, alt or .vcf."
        )

    df_pred = predict_variant_effect(
        df_variant,
        tasks=tasks,
        model=model,
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
        min_from_end=min_from_end,
        max_dist_tss=max_dist_tss,
        include_cols=include_cols,
    )
    df_pred.to_parquet(output_pq)
