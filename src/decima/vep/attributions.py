"""
Variant Effect Attribution Module.

This module provides functionality to compute feature attributions for genetic variants.
It calculates the contribution of input sequences to model predictions, allowing for the
interpretation of variant effects in motifs of transcription factors.

Examples:
    >>> variant_effect_attribution(
    ...     df_variant="variants.vcf",
    ...     output_h5="attributions.h5",
    ...     tasks=[
    ...         "T_cell",
    ...         "B_cell",
    ...     ],
    ...     model=0,
    ...     metadata_anndata="results.h5ad",
    ... )
"""

import logging
from typing import Optional, Union, List
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm

from decima.constants import ENSEMBLE_MODELS, MODEL_METADATA
from decima.utils import get_compute_device, _get_on_off_tasks
from decima.utils.io import read_vcf_chunks, VariantAttributionWriter
from decima.core.result import DecimaResult
from decima.data.dataset import VariantDataset
from decima.interpret.attributer import DecimaAttributer
from decima.model.metrics import WarningCounter
from decima.vep.vep import _log_vep_warnings, _write_vep_warnings


def variant_effect_attribution(
    variants: Union[pd.DataFrame, str],
    output_prefix: str,
    tasks: Optional[Union[str, List[str]]] = None,
    off_tasks: Optional[Union[str, List[str]]] = None,
    model: int = 0,
    metadata_anndata: Optional[str] = None,
    method: str = "inputxgradient",
    transform: str = "specificity",
    batch_size: int = 1,
    num_workers: int = 4,
    device: Optional[str] = None,
    gene_col: Optional[str] = None,
    distance_type: Optional[str] = "tss",
    min_distance: Optional[float] = 0,
    max_distance: Optional[float] = float("inf"),
    genome: str = "hg38",
):
    """
    Computes variant effect attributions for a set of variants and writes them to an HDF5 file.

    This function calculates the contribution of input features (sequence) to the model's
    prediction for specific tasks (cell types), contrasting with off-target tasks if specified.
    It supports various attribution methods (e.g., InputXGradient) and transformations
    (e.g., Specificity).

    Args:
        df_variant (Union[pd.DataFrame, str]): Input variants. Can be a pandas DataFrame or a path
            to a file (.tsv, .csv, or .vcf). If a file path is provided, it will be loaded.
            Required columns/fields depend on the input format but generally include chromosome,
            position, reference allele, alternate allele.
        output_prefix (str): Path to the output HDF5 file where attributions will be saved.
            If None, results might not be persisted.
        tasks (Union[str, List[str]], optional): Specific task(s) or cell type(s) to compute
            attributions for. If None, uses all available tasks or a default set. Defaults to None.
        off_tasks (Union[str, List[str]], optional): Task(s) to use as a background or negative
            set for specificity calculations. Defaults to None.
        model (int, optional): Index or identifier of the model to use from the ensemble.
            Defaults to 0.
        metadata_anndata (str, optional): Path to the AnnData file containing model metadata and
            results (DecimaResult). Used to resolve task names and indices. Defaults to None.
        method (str, optional): The attribution method to use. Options: "inputxgradient",
            "saliency", "integratedgradients". Defaults to "inputxgradient".
        transform (str, optional): Transformation to apply to the model output before attribution.
            Options: "specificity" (target - off_target) or "aggregate". Defaults to "specificity".
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 4.
        device (str, optional): Compute device to use (e.g., "cpu", "cuda", "cuda:0").
            If None, automatically detects available device. Defaults to None.
        gene_col (str, optional): Name of the column in `df_variant` containing gene identifiers.
            If provided, variants are associated with specific genes. Defaults to None.
        distance_type (str, optional): Method to calculate distance between variant and gene.
            Options: "tss" (Transcription Start Site). Defaults to "tss".
        min_distance (float, optional): Minimum distance from the gene feature (e.g., TSS) for a
            variant to be included. Defaults to 0.
        max_distance (float, optional): Maximum distance from the gene feature (e.g., TSS) for a
            variant to be included. Defaults to infinity.
        genome (str, optional): Genome assembly version (e.g., "hg38"). Defaults to "hg38".

    Returns:
        List[str]: List of paths to the output HDF5 files.

    Examples:
        Compute attributions for variants in a VCF file for specific tasks:

        >>> variant_effect_attribution(
        ...     variants="variants.vcf",
        ...     output_prefix="attributions",
        ...     tasks=[
        ...         "T_cell",
        ...         "B_cell",
        ...     ],
        ...     model=0,
        ...     metadata_anndata="results.h5ad",
        ... )
    """
    if (model in ENSEMBLE_MODELS) or isinstance(model, (list, tuple)):
        if model in ENSEMBLE_MODELS:
            models = MODEL_METADATA[model]
        else:
            models = model
        return [
            variant_effect_attribution(
                variants=variants,
                output_prefix=(str(output_prefix) + "_{model}").format(model=idx),
                tasks=tasks,
                off_tasks=off_tasks,
                model=model,
                metadata_anndata=metadata_anndata,
                method=method,
                transform=transform,
                batch_size=batch_size,
                num_workers=num_workers,
                device=device,
                gene_col=gene_col,
                distance_type=distance_type,
                min_distance=min_distance,
                max_distance=max_distance,
                genome=genome,
            )
            for idx, model in enumerate(models)
        ]

    logger = logging.getLogger("decima")
    device = get_compute_device(device)
    logger.info(f"Using device: {device} and genome: {genome}")

    if isinstance(variants, str):
        if variants.endswith(".tsv"):
            variants = pd.read_csv(variants, sep="\t")
        elif variants.endswith(".csv"):
            variants = pd.read_csv(variants, sep=",")
        elif variants.endswith(".vcf") or variants.endswith(".vcf.gz"):
            variants = next(read_vcf_chunks(variants, chunksize=float("inf")))
        else:
            raise ValueError(f"Unsupported file extension: {variants}. Must be .tsv or .vcf.")
    elif isinstance(variants, pd.DataFrame):
        pass
    else:
        raise ValueError(
            f"Unsupported input type: {type(variants)}. Must be pd.DataFrame or str (path to .tsv or .vcf)."
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
        variants,
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
    dl = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=dataset.collate_fn
    )

    seqs = list()
    attrs = list()
    for batch in tqdm(dl, total=len(dataset), desc="Computing attributions..."):
        seqs.append(batch["seq"].cpu().numpy())
        attrs.append(attributer.attribute(batch["seq"].to(device)).detach().cpu().float().numpy())
        warning_counter.update(batch["warning"])

    Path(output_prefix).parent.mkdir(parents=True, exist_ok=True)

    output_h5 = str(output_prefix) + ".h5"
    with VariantAttributionWriter(
        path=output_h5,
        genes=genes,
        variants=variants,
        model_name=attributer.model.name,
        metadata_anndata=result,
        genome=genome,
    ) as writer:
        for variant, gene, rel_pos, gene_mask_start, gene_mask_end, seqs_ref, seqs_alt, attrs_ref, attrs_alt in tqdm(
            zip(
                variants,
                genes,
                dataset.variants["rel_pos"],
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
                rel_pos=rel_pos,
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

    return output_h5
