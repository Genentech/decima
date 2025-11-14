"""Attributions module predict attributes and performs recursive seqlet calling.

Examples:
    >>> predict_save_attributions(
    ...     output_prefix="output_prefix",
    ...     tasks=[
    ...         "agg1",
    ...         "agg2",
    ...     ],
    ...     off_tasks=[
    ...         "agg3",
    ...         "agg4",
    ...     ],
    ... )
    >>> recursive_seqlet_calling(
    ...     output_prefix="output_prefix",
    ...     attributions="attributions.h5",
    ...     tasks=[
    ...         "agg1",
    ...         "agg2",
    ...     ],
    ...     off_tasks=[
    ...         "agg3",
    ...         "agg4",
    ...     ],
    ... )
"""

import glob
import warnings
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from more_itertools import chunked
from torch.utils.data import DataLoader
from pyfaidx import Faidx

from decima.constants import DEFAULT_ENSEMBLE, MODEL_METADATA, ENSEMBLE_MODELS
from decima.core.attribution import AttributionResult
from decima.core.result import DecimaResult
from decima.data.dataset import GeneDataset, SeqDataset
from decima.hub import load_decima_model
from decima.interpret.attributer import DecimaAttributer
from decima.utils import get_compute_device, _get_on_off_tasks, _get_genes
from decima.utils.io import AttributionWriter
from decima.utils.qc import QCLogger


def predict_save_attributions(
    output_prefix: str,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[int] = DEFAULT_ENSEMBLE,
    metadata_anndata: Optional[str] = None,
    method: str = "inputxgradient",
    transform: str = "specificity",
    batch_size: int = 1,
    genes: Optional[List[str]] = None,
    seqs: Optional[Union[str, pd.DataFrame, np.ndarray, torch.Tensor]] = None,
    top_n_markers: Optional[int] = None,
    bigwig: bool = True,
    correct_grad_bigwig: bool = True,
    num_workers: int = 4,
    device: Optional[str] = None,
    genome: str = "hg38",
):
    """Generate and save attribution analysis results for a gene.

    Args:
        output_prefix: Prefix for the output files where attribution results will be saved.
        tasks: Tasks to attribute for prediction either list of task names or query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte''). If not provided, all tasks will be analyzed.
        off_tasks: Off tasks to attribute for prediction either list of task names or query string to filter cell types to contrast against (e.g. 'cell_type == 'classical monocyte''). If not provided, no contrast will be performed.
        model: Model to attribute for prediction default is 0. Can be replicate number (0-3) or path to custom model.
        metadata_anndata: Metadata anndata path or DecimaResult object. If not provided, the default metadata will be downloaded from wandb.
        method: Method to use for attribution analysis available options: "saliency", "inputxgradient", "integratedgradients". Default is "inputxgradient".
        transform: Transform to use for attribution analysis available options: "specificity", "aggregate". Default is "specificity".
        batch_size: Batch size for attribution analysis default is 1. Increasing batch size may speed up computation but requires more memory.
        genes: Genes to attribute for prediction if not provided, all genes will be used. Can be list of gene symbols or IDs.
        seqs: Sequences to attribute for prediction. Can be path to fasta file, DataFrame, or numpy/torch tensor. Mutually exclusive with genes parameter.
        top_n_markers: Top n markers for prediction if not provided, genes will be used. Useful for analyzing only the most important marker genes.
        bigwig: Whether to save attribution scores as a bigwig file default is True. Bigwig files can be loaded in genome browsers for visualization.
        correct_grad_bigwig: Whether to correct the gradient bigwig file default is True. Applies gradient correction for better visualization.
        num_workers: Number of workers for attribution analysis default is 4. Increasing number of workers will speed up the process.
        device: Device to use for attribution analysis (e.g. 'cuda', 'cpu'). If not provided, the best available device will be used automatically.
        genome: Genome to use for attribution analysis default is "hg38". Can be genome name or path to custom genome fasta file.

    Returns:
        Path to the attribution file.

    Examples:
        >>> predict_save_attributions(
        ...     output_prefix="output_prefix",
        ...     tasks=[
        ...         "task1",
        ...         "task2",
        ...     ],
        ...     off_tasks=[
        ...         "task3",
        ...         "task4",
        ...     ],
        ...     model=0,
        ...     metadata_anndata="metadata_anndata.h5ad",
        ...     method="inputxgradient",
        ...     transform="specificity",
        ...     batch_size=1,
        ...     genes=[
        ...         "gene1",
        ...         "gene2",
        ...     ],
        ...     seqs="seqs.fasta",
        ...     top_n_markers=10,
        ...     bigwig=True,
        ...     correct_grad_bigwig=True,
        ...     num_workers=4,
        ...     device="cuda",
        ...     genome="hg38",
        ... )
    """
    if (model in ENSEMBLE_MODELS) or isinstance(model, (list, tuple)):
        if model in ENSEMBLE_MODELS:
            models = MODEL_METADATA[model]
        else:
            models = model
        return [
            predict_save_attributions(
                output_prefix=(str(output_prefix) + "_{model}").format(model=idx),
                tasks=tasks,
                off_tasks=off_tasks,
                model=model,
                metadata_anndata=metadata_anndata,
                method=method,
                transform=transform,
                batch_size=batch_size,
                genes=genes,
                seqs=seqs,
                top_n_markers=top_n_markers,
                bigwig=bigwig,
                correct_grad_bigwig=correct_grad_bigwig,
                num_workers=num_workers,
                device=device,
                genome=genome,
            )
            for idx, model in enumerate(models)
        ]

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("decima")

    device = get_compute_device(device)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model {model} and metadata to compute attributions...")
    model = load_decima_model(model, device=device)
    result = DecimaResult.load(metadata_anndata or model.name)

    tasks, off_tasks = _get_on_off_tasks(result, tasks, off_tasks)
    attributer = DecimaAttributer(model, tasks, off_tasks, method, transform)

    with QCLogger(str(output_prefix) + ".warnings.qc.log", metadata_anndata=result) as qc:
        if result.ground_truth is not None:
            qc.log_correlation(tasks, off_tasks, plot=True)

        if (genes is not None) and (seqs is not None):
            raise ValueError("Only one of `genes` or `seqs` arguments must be provided not both.")
        elif seqs is not None:
            assert top_n_markers is None, "`top_n_markers` is not supported when `seqs` is provided."

            if isinstance(seqs, str):
                dataset = SeqDataset.from_fasta(seqs)
            elif isinstance(seqs, pd.DataFrame):
                dataset = SeqDataset.from_dataframe(seqs)
            elif isinstance(seqs, torch.Tensor) or isinstance(seqs, np.ndarray):
                assert seqs.shape[1] == 5, (
                    "`seqs` must be 5-dimensional with shape (batch_size, 5, seq_len) "
                    "where the 2th dimension is a one_hot encoded seq and binary mask gene mask."
                )
                dataset = SeqDataset.from_one_hot(seqs)
            else:
                raise ValueError(f"Invalid type for seqs: {type(seqs)}. Must be a path to fasta file or pd.DataFrame.")
        else:
            dataset = GeneDataset(
                genes=_get_genes(result, genes, top_n_markers, tasks, off_tasks), metadata_anndata=result, genome=genome
            )

        genes_batch = list(chunked(dataset.genes, batch_size))
        gene_mask_starts = list(chunked(dataset.gene_mask_starts, batch_size))
        gene_mask_ends = list(chunked(dataset.gene_mask_ends, batch_size))

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        output_path = Path(output_prefix).with_suffix(".attributions.h5")
        with AttributionWriter(
            path=output_path,
            genes=dataset.genes,
            model_name=attributer.model.name,
            metadata_anndata=result,
            genome=genome,
            bigwig=bigwig,
            correct_grad_bigwig=correct_grad_bigwig,
            custom_genes=seqs is not None,
        ) as writer:
            for i, inputs in enumerate(tqdm(dl, desc="Computing attributions...")):
                attrs = attributer.attribute(inputs.to(device)).detach().cpu().numpy()
                _seqs = inputs[:, :4].detach().cpu().numpy()

                for gene, seq, attr, g_start, g_end in zip(
                    genes_batch[i], _seqs, attrs, gene_mask_starts[i], gene_mask_ends[i]
                ):
                    writer.add(gene=gene, seqs=seq, attrs=attr, gene_mask_start=g_start, gene_mask_end=g_end)
                    if seqs is None:
                        qc.log_gene(gene, threshold=0.5)

        if seqs is not None:
            logger.info("Saving sequences...")
            fasta_path = str(Path(output_prefix).with_suffix(".seqs.fasta"))
            with open(fasta_path, "w") as f:
                for i, seq in tqdm(zip(dataset.genes, dataset.seqs), desc="Saving sequences..."):
                    f.write(f">{i}\n{seq}\n")
            Faidx(fasta_path, build_index=True)

        return output_path


def recursive_seqlet_calling(
    output_prefix: str,
    attributions: Union[str, List[str]],
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    tss_distance: Optional[int] = None,
    metadata_anndata: Optional[str] = None,
    genes: Optional[List[str]] = None,
    top_n_markers: Optional[int] = None,
    num_workers: int = 4,
    agg_func: Optional[str] = "mean",
    threshold: float = 5e-4,
    min_seqlet_len: int = 4,
    max_seqlet_len: int = 25,
    additional_flanks: int = 0,
    pattern_type: str = "both",
    custom_genome: bool = False,
    meme_motif_db: str = "hocomoco_v13",
):
    """Recursive seqlet calling for attribution analysis.

    Args:
        output_prefix: Prefix for the output files where seqlet calling results will be saved.
        attributions: Attributions to use for recursive seqlet calling generated by `decima attributions-predict` or `decima attributions` commands. Can be single file path or list of attribution files.
        tasks: Tasks to attribute for recursive seqlet calling either list of task names or query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte''). If not provided, all tasks will be analyzed.
        off_tasks: Off tasks to attribute for recursive seqlet calling either list of task names or query string to filter cell types to contrast against (e.g. 'cell_type == 'classical monocyte''). If not provided, no contrast will be performed.
        tss_distance: TSS distance for recursive seqlet calling default is full context size of decima of 524288. Controls the genomic window size around TSS for analysis.
        metadata_anndata: Metadata anndata path or DecimaResult object. If not provided, the default metadata will be used from the attribution files.
        genes: Genes to attribute for recursive seqlet calling if not provided, all genes will be used. Can be list of gene symbols or IDs to focus analysis on specific genes.
        top_n_markers: Top n markers for recursive seqlet calling if not provided, genes will be used. Useful for analyzing only the most important marker genes for the specified tasks.
        num_workers: Number of workers for recursive seqlet calling default is 4. Increasing number of workers will speed up the process but requires more memory.
        agg_func: Aggregation function for recursive seqlet calling default is 'mean'. Available options: 'mean', 'max'. Determines how attribution scores are aggregated across replicates.
        threshold: P-value threshold for recursive seqlet calling default is 5e-4. Lower values result in more stringent peak calling and fewer detected seqlets.
        min_seqlet_len: Minimum seqlet length for recursive seqlet calling default is 4. Shorter sequences will be filtered out from the analysis.
        max_seqlet_len: Maximum seqlet length for recursive seqlet calling default is 25. Longer sequences will be truncated or filtered based on the algorithm.
        additional_flanks: Additional flanks for recursive seqlet calling default is 0. Extends seqlet regions by this number of base pairs on each side.
        pattern_type: Pattern type for recursive seqlet calling default is 'both'. Available options: 'both', 'pos', 'neg'. Controls whether to consider positive peaks, negative peaks, or both.
        custom_genome: Custom genome flag for recursive seqlet calling default is False. If True, bigwig files will be generated with each gene as a different chromosome for custom sequences.
        meme_motif_db: MEME motif database for motif discovery default is 'hocomoco_v13'. Specifies which motif database to use for downstream motif enrichment analysis.

    Examples:
        >>> recursive_seqlet_calling(
        ...     output_prefix="output_prefix",
        ...     attributions="attributions.h5",
        ...     tasks=[
        ...         "task1",
        ...         "task2",
        ...     ],
        ...     off_tasks=[
        ...         "task3",
        ...         "task4",
        ...     ],
        ... )
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # TODO: update dependencies so we do not get future errors.
    warnings.filterwarnings("ignore", category=FutureWarning)

    logger = logging.getLogger("decima")
    logger.info("Loading model and metadata to compute attributions...")

    if isinstance(attributions, (str, Path)):
        attributions_files = [Path(attributions).as_posix()]
    else:
        attributions_files = attributions

    with AttributionResult(
        attributions_files, tss_distance, correct_grad=False, num_workers=num_workers, agg_func=agg_func
    ) as ar:
        result = DecimaResult.load(metadata_anndata or ar.model_name)

        if top_n_markers is not None:
            tasks, off_tasks = _get_on_off_tasks(result, tasks, off_tasks)
            all_genes = _get_genes(result, genes, top_n_markers, tasks, off_tasks)
        elif genes is not None:
            all_genes = genes if isinstance(genes, list) else [genes]
        else:
            all_genes = ar.genes
            logger.info(f"No genes provided, using all {len(all_genes)} genes in the attribution files.")

        df_peaks, df_motifs = ar.recursive_seqlet_calling(
            all_genes,
            metadata_anndata,
            custom_genome,
            threshold,
            min_seqlet_len,
            max_seqlet_len,
            additional_flanks,
            pattern_type,
            meme_motif_db,
        )

    df_peaks.sort_values(["chrom", "start"]).to_csv(
        output_prefix.with_suffix(".seqlets.bed"), sep="\t", header=False, index=False
    )
    df_motifs.to_csv(output_prefix.with_suffix(".motifs.tsv"), sep="\t", index=False)


def predict_attributions_seqlet_calling(
    output_prefix: str,
    genes: Optional[Union[str, List[str]]] = None,
    seqs: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[Union[str, int]] = DEFAULT_ENSEMBLE,
    metadata_anndata: Optional[str] = None,
    method: str = "inputxgradient",
    transform: str = "specificity",
    num_workers: int = 2,
    tss_distance: Optional[int] = None,
    batch_size: int = 1,
    top_n_markers: Optional[int] = None,
    device: Optional[str] = None,
    threshold: float = 5e-4,
    min_seqlet_len: int = 4,
    max_seqlet_len: int = 25,
    additional_flanks: int = 0,
    pattern_type: str = "both",
    meme_motif_db: str = "hocomoco_v13",
    genome: str = "hg38",
):
    """Generate and save attribution analysis results for a gene.
    This function performs attribution analysis for a given gene and cell types, saving
    the following output files to the specified directory:

    Output files:
        ├── {output_prefix}.attributions.h5      # Raw attribution score matrix per gene.

        ├── {output_prefix}.attributions.bigwig  # Genome browser track of attribution as bigwig file.

        ├── {output_prefix}.seqlets.bed          # List of attribution peaks in BED format.

        ├── {output_prefix}.motifs.tsv           # Detected motifs in peak regions.

        └── {output_prefix}.warnings.qc.log      # QC warnings about prediction reliability.

    Args:
        output_dir: Directory to save output files
        gene: Gene symbol or ID to analyze
        tasks: List of cell types to analyze attributions for either list of task names or query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte''). If not provided, all tasks will be analyzed.
        off_tasks: Optional list of cell types to contrast against either list of task names or query string to filter cell types to contrast against (e.g. 'cell_type == 'classical monocyte''). If not provided, all tasks will be used as off tasks.
        model: Optional model to use for attribution analysis either replicate number or path to the model.
        method: Method to use for attribution analysis available options: "saliency", "inputxgradient", "integratedgradients".
        device: Device to use for attribution analysis (e.g. 'cuda', 'cpu'). If not provided, the best available device will be used automatically.
        dpi: DPI for attribution plots.

    Raises:
        FileExistsError: If output directory already exists.

    Examples:
    >>> predict_save_attributions(
    ...     output_dir="output_dir",
    ...     genes=[
    ...         "SPI1",
    ...         "CD68",
    ...     ],
    ...     tasks="cell_type == 'classical monocyte'",
    ... )
    """
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    attributions_paths = predict_save_attributions(
        output_prefix=output_prefix,
        genes=genes,
        seqs=seqs,
        tasks=tasks,
        off_tasks=off_tasks,
        model=model,
        metadata_anndata=metadata_anndata,
        method=method,
        transform=transform,
        num_workers=num_workers,
        batch_size=batch_size,
        top_n_markers=top_n_markers,
        device=device,
        genome=genome,
    )

    custom_genome = False
    if seqs is not None:
        custom_genome = True
        assert genes is None, "`genes` must be None when `seqs` is provided."

    recursive_seqlet_calling(
        output_prefix=output_prefix,
        attributions=attributions_paths,
        metadata_anndata=metadata_anndata,
        genes=genes,
        tasks=tasks,
        off_tasks=off_tasks,
        tss_distance=tss_distance,
        num_workers=num_workers,
        custom_genome=custom_genome,
        threshold=threshold,
        min_seqlet_len=min_seqlet_len,
        max_seqlet_len=max_seqlet_len,
        additional_flanks=additional_flanks,
        pattern_type=pattern_type,
        meme_motif_db=meme_motif_db,
    )


def plot_attributions(
    output_prefix: str,
    genes: Optional[Union[str, List[str]]] = None,
    metadata_anndata: Optional[str] = None,
    tss_distance: Optional[int] = None,
    seqlogo_window: int = 50,
    agg_func: Optional[str] = "mean",
    custom_genome: bool = False,
    dpi: int = 100,
):
    """Plot attributions.

    Args:
        output_prefix: Prefix for the output files.
        genes: Genes to attribute if not provided, all genes will be used.
        metadata_anndata: Metadata anndata.
        tss_distance: TSS distance for attribution for plotting.
        seqlogo_window: Seqlogo window.
        agg_func: Agg func for aggregation of attributions across replicates. Available options: 'mean', 'max'.
        custom_genome: Custom genome if custom genome bigwig files will be generated as each gene is difference chromosome.
        dpi: DPI for attribution plots.
    """
    plot_dir = Path(str(output_prefix) + "_plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    with AttributionResult(
        glob.glob(str(output_prefix) + "*.attributions.h5"), tss_distance, correct_grad=False, agg_func=agg_func
    ) as ar:
        # TODO: if we save seqlets as h5 we do not need to recalculate them
        # TODO: create html reports
        for gene in tqdm(genes, desc="Plotting attributions..."):
            attribution = ar.load_attribution(gene, metadata_anndata, custom_genome)
            attribution.plot_peaks().save((plot_dir / f"{gene}.peaks.png"), dpi=dpi)

            seqlogo_plot_dir = plot_dir / f"{gene}_seqlogos"
            seqlogo_plot_dir.mkdir(parents=True, exist_ok=True)
            for peak in attribution.peaks.itertuples():
                logo = attribution.plot_seqlogo(relative_loc=peak.from_tss, window=seqlogo_window)
                logo.ax.figure.savefig(seqlogo_plot_dir / f"{attribution.gene}@{peak.from_tss}.png", dpi=dpi)
