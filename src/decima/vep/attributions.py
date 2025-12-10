import logging
from typing import Optional, Union, List

import torch
import pandas as pd
from tqdm import tqdm

from decima.utils import get_compute_device, _get_on_off_tasks
from decima.utils.io import read_vcf_chunks, VariantAttributionWriter
from decima.core.result import DecimaResult
from decima.data.dataset import VariantDataset
from decima.interpret.attributer import DecimaAttributer
from decima.model.metrics import WarningCounter
from decima.vep.vep import _log_vep_warnings, _write_vep_warnings


def variant_effect_attribution(
    df_variant: Union[pd.DataFrame, str],
    output_h5: Optional[str] = None,
    tasks: Optional[Union[str, List[str]]] = None,
    off_tasks: Optional[Union[str, List[str]]] = None,
    model: int = 0,
    metadata_anndata: Optional[str] = None,
    method: str = "inputxgradient",
    transform: str = "specificity",
    num_workers: int = 4,
    device: Optional[str] = None,
    gene_col: Optional[str] = None,
    distance_type: Optional[str] = "tss",
    min_distance: Optional[float] = 0,
    max_distance: Optional[float] = float("inf"),
    genome: str = "hg38",
    float_precision: str = "32",
):
    """ """

    logger = logging.getLogger("decima")
    device = get_compute_device(device)
    logger.info(f"Using device: {device} and genome: {genome}")

    if isinstance(df_variant, pd.DataFrame):
        pass
    elif isinstance(df_variant, str):
        if df_variant.endswith(".tsv"):
            df_variant = pd.read_csv(df_variant, sep="\t")
        elif df_variant.endswith(".csv"):
            df_variant = pd.read_csv(df_variant, sep=",")
        elif df_variant.endswith(".vcf") or df_variant.endswith(".vcf.gz"):
            df_variant = next(read_vcf_chunks(df_variant, chunksize=float("inf")))
        else:
            raise ValueError(f"Unsupported file extension: {df_variant}. Must be .tsv or .vcf.")
    else:
        raise ValueError(
            f"Unsupported input type: {type(df_variant)}. Must be pd.DataFrame or str (path to .tsv or .vcf)."
        )

    result = DecimaResult.load(metadata_anndata)

    tasks, off_tasks = _get_on_off_tasks(result, tasks, off_tasks)
    attributer = DecimaAttributer.load_decima_attributer(
        model_name=model,
        tasks=tasks,
        off_tasks=off_tasks,
        method=method,
        transform=transform,
        device=device,
    )

    warning_counter = WarningCounter()

    dataset = VariantDataset(
        df_variant,
        metadata_anndata=metadata_anndata,
        gene_col=gene_col,
        distance_type=distance_type,
        min_distance=min_distance,
        max_distance=max_distance,
        reference_cache=False,
        genome=genome,
    )
    variants = (
        dataset.variants["chrom"]
        + "_"
        + dataset.variants["pos"].astype(str)
        + "_"
        + dataset.variants["ref"]
        + "_"
        + dataset.variants["alt"]
    )
    genes = dataset.variants["gene"]
    dl = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=num_workers, collate_fn=dataset.collate_fn)

    seqs = list()
    attrs = list()
    for batch in tqdm(dl, total=len(dataset), desc="Computing attributions..."):
        seqs.append(batch["seq"].cpu().numpy())
        attrs.append(attributer.attribute(batch["seq"].to(device)).detach().cpu().float().numpy())
        warning_counter.update(batch["warning"])

    with VariantAttributionWriter(
        path=output_h5,
        genes=genes,
        variants=variants,
        model_name=attributer.model.name,
        metadata_anndata=result,
        genome=genome,
    ) as writer:
        for variant, gene, gene_mask_start, gene_mask_end, seqs_ref, seqs_alt, attrs_ref, attrs_alt in tqdm(
            zip(
                variants,
                genes,
                dataset.variants["gene_mask_start"],
                dataset.variants["gene_mask_end"],
                seqs[::2],
                seqs[1::2],
                attrs[::2],
                attrs[1::2],
            ),
            total=len(variants),
            desc="Writing attributions...",
        ):
            writer.add(
                variant=variant,
                gene=gene,
                seqs_ref=seqs_ref[0, :4],
                seqs_alt=seqs_alt[0, :4],
                attrs_ref=attrs_ref[0, :4],
                attrs_alt=attrs_alt[0, :4],
                gene_mask_start=gene_mask_start,
                gene_mask_end=gene_mask_end,
            )

    warning_counter = warning_counter.compute()
    _log_vep_warnings(warning_counter, len(variants), genome)
    _write_vep_warnings(warning_counter, len(variants), output_h5)
