import warnings
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import modiscolite
from tqdm import tqdm
from more_itertools import chunked
from torch.utils.data import DataLoader
from grelu.resources import get_meme_file_path


from decima.constants import DECIMA_CONTEXT_SIZE
from decima.core.result import DecimaResult
from decima.data.dataset import GeneDataset
from decima.utils import get_compute_device
from decima.utils.io import AttributionWriter
from decima.core.attribution import AttributionResult
from decima.interpret.attributer import DecimaAttributer


def _get_on_off_tasks(result: DecimaResult, tasks: Optional[List[str]] = None, off_tasks: Optional[List[str]] = None):
    if tasks is None:
        tasks = result.cell_metadata.index.tolist()
    elif isinstance(tasks, str):
        tasks = result.query_cells(tasks)
    if isinstance(off_tasks, str):
        off_tasks = result.query_cells(off_tasks)

    return tasks, off_tasks


def _get_genes(
    result: DecimaResult,
    genes: Optional[List[str]] = None,
    top_n_markers: Optional[int] = None,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
):
    if (top_n_markers is not None) and (genes is None):
        all_genes = (
            result.marker_zscores(tasks=tasks, off_tasks=off_tasks)
            .query('task == "on"')
            .sort_values("score", ascending=False)
            .drop_duplicates(subset="gene", keep="first")
            .iloc[:top_n_markers]
            .gene.tolist()
        )
    elif genes is not None:
        if top_n_markers is not None:
            raise ValueError(
                "Cannot specify arguments `genes` and `top_n_markers` at the same time. Only one can be specified."
            )
        all_genes = genes
    else:
        all_genes = list(result.genes)

    return all_genes


def predict_save_modisco_attributions(
    output_prefix: str,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[int] = 0,
    metadata_anndata: Optional[str] = None,
    method: str = "saliency",
    transform: str = "specificity",
    batch_size: int = 2,
    genes: Optional[List[str]] = None,
    top_n_markers: Optional[int] = None,
    num_workers: int = 4,
    device: Optional[str] = None,
    genome: str = "hg38",
):
    """Generate and save attribution analysis results for a gene.
    This function performs attribution analysis for a given gene and cell types, saving
    the following output files to the specified directory:

    Args:
        output_prefix: Path to save attribution scores.
        tasks: List of cell types to analyze attributions for.
        off_tasks: Optional list of cell types to contrast against.
        model: Optional model to use for attribution analysis.
        metadata_anndata: Path to the metadata anndata file.
        method: Method to use for attribution analysis.
        chunk_size: Chunk size for the prediction.
        batch_size: Batch size for the prediction.
        genes: List of genes to analyze attributions for.
        top_n_markers: Top n markers to predict. If not provided, all markers will be predicted.
        num_workers: Number of workers for the prediction.
        device: Device to use for attribution analysis.
        genome: Genome name or path to the genome fasta file.

    Raises:
        FileExistsError: If output directory already exists.

    Examples:
    >>> predict_save_attributions(
    ...     output_dir="output_dir",
    ...     tasks="cell_type == 'classical monocyte'",
    ... )
    """
    warnings.filterwarnings("ignore", category=FutureWarning, module="tangermeme")

    # TODO: QC how well model predicts on tasks from on tasks
    logger = logging.getLogger("decima")

    device = get_compute_device(device)
    logger.info(f"Using device: {device}")

    logger.info("Loading model and metadata to compute attributions...")
    result = DecimaResult.load(metadata_anndata)

    tasks, off_tasks = _get_on_off_tasks(result, tasks, off_tasks)
    all_genes = _get_genes(result, genes, top_n_markers, tasks, off_tasks)

    attributer = DecimaAttributer.load_decima_attributer(model, tasks, off_tasks, method, transform, device=device)

    dl = DataLoader(
        GeneDataset(genes=all_genes, metadata_anndata=result),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )
    genes = list(chunked(all_genes, batch_size))

    with AttributionWriter(
        path=Path(output_prefix).with_suffix(".attributions.h5"),
        genes=all_genes,
        model_name=attributer.model.name,
        metadata_anndata=result,
        genome=genome,
        bigwig=True,
    ) as writer:
        for i, inputs in enumerate(tqdm(dl, desc="Computing attributions...")):
            attrs = attributer.attribute(inputs.to(device)).detach().cpu().numpy()
            seqs = inputs[:, :4].detach().cpu().numpy()

            for gene, attr, seq in zip(genes[i], attrs, seqs):
                writer.add(gene=gene, seqs=seq, attrs=attr)


def modisco_patterns(
    output_prefix: str,
    attributions: Union[str, List[str]],
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    tss_distance: int = 10_000,
    metadata_anndata: Optional[str] = None,
    genes: Optional[List[str]] = None,
    top_n_markers: Optional[int] = None,
    correct_grad: bool = True,
    num_workers: int = 4,
    # tfmodisco parameters
    sliding_window_size: int = 20,
    flank_size: int = 10,
    min_metacluster_size: int = 100,
    weak_threshold_for_counting_sign: float = 0.8,
    max_seqlets_per_metacluster: int = 20_000,
    target_seqlet_fdr: float = 0.2,
    min_passing_windows_frac: float = 0.03,
    max_passing_windows_frac: float = 0.2,
    n_leiden_runs: int = 16,
    n_leiden_iterations: int = -1,
    min_overlap_while_sliding: float = 0.7,
    nearest_neighbors_to_compute: int = 500,
    affmat_correlation_threshold: float = 0.15,
    tsne_perplexity: float = 10.0,
    frac_support_to_trim_to: float = 0.2,
    min_num_to_trim_to: int = 30,
    trim_to_window_size: int = 30,
    initial_flank_to_add: int = 10,
    final_flank_to_add: int = 0,
    prob_and_pertrack_sim_merge_thresholds: List[Tuple[float, float]] = [(0.8, 0.8), (0.5, 0.85), (0.2, 0.9)],
    prob_and_pertrack_sim_dealbreaker_thresholds: List[Tuple[float, float]] = [
        (0.4, 0.75),
        (0.2, 0.8),
        (0.1, 0.85),
        (0.0, 0.9),
    ],
    subcluster_perplexity: int = 50,
    merging_max_seqlets_subsample: int = 300,
    final_min_cluster_size: int = 20,
    min_ic_in_window: float = 0.6,
    min_ic_windowsize: int = 6,
    ppm_pseudocount: float = 0.001,
    stranded: bool = False,
    pattern_type: str = "both",  # "both", "pos", or "neg"
):
    logger = logging.getLogger("decima")
    logger.info("Loading metadata")
    result = DecimaResult.load(metadata_anndata)

    if isinstance(attributions, (str, Path)):
        attributions_files = [Path(attributions).as_posix()]
    else:
        attributions_files = attributions

    tasks, off_tasks = _get_on_off_tasks(result, tasks, off_tasks)
    all_genes = _get_genes(result, genes, top_n_markers, tasks, off_tasks)

    sequences = list()
    attributions = list()

    for attributions_file in tqdm(attributions_files, desc="Loading attributions and sequences..."):
        with AttributionResult(attributions_file, metadata_anndata, tss_distance, correct_grad) as attributions_result:
            seqs, attrs = attributions_result.load(all_genes)
            sequences.append(seqs)
            attributions.append(attrs)

    sequences = np.mean(sequences, axis=0)
    attributions = np.mean(attributions, axis=0)

    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        hypothetical_contribs=attributions.transpose(0, 2, 1),
        one_hot=sequences.transpose(0, 2, 1),
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
        min_metacluster_size=min_metacluster_size,
        weak_threshold_for_counting_sign=weak_threshold_for_counting_sign,
        max_seqlets_per_metacluster=max_seqlets_per_metacluster,
        target_seqlet_fdr=target_seqlet_fdr,
        min_passing_windows_frac=min_passing_windows_frac,
        max_passing_windows_frac=max_passing_windows_frac,
        n_leiden_runs=n_leiden_runs,
        n_leiden_iterations=n_leiden_iterations,
        min_overlap_while_sliding=min_overlap_while_sliding,
        nearest_neighbors_to_compute=nearest_neighbors_to_compute,
        affmat_correlation_threshold=affmat_correlation_threshold,
        tsne_perplexity=tsne_perplexity,
        frac_support_to_trim_to=frac_support_to_trim_to,
        min_num_to_trim_to=min_num_to_trim_to,
        trim_to_window_size=trim_to_window_size,
        initial_flank_to_add=initial_flank_to_add,
        final_flank_to_add=final_flank_to_add,
        prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
        prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
        subcluster_perplexity=subcluster_perplexity,
        merging_max_seqlets_subsample=merging_max_seqlets_subsample,
        final_min_cluster_size=final_min_cluster_size,
        min_ic_in_window=min_ic_in_window,
        min_ic_windowsize=min_ic_windowsize,
        ppm_pseudocount=ppm_pseudocount,
        stranded=stranded,
        pattern_type=pattern_type,
        num_cores=num_workers,
        verbose=True,
    )
    modiscolite.io.save_hdf5(
        Path(output_prefix).with_suffix(".modisco.h5").as_posix(),
        pos_patterns,
        neg_patterns,
        window_size=tss_distance * 2 if tss_distance is not None else DECIMA_CONTEXT_SIZE,
    )


def modisco_reports(
    output_prefix: str,
    modisco_h5: str,
    meme_motif_db: Optional[Union[Path, str]] = "hocomoco_v13",
    img_path_suffix: Optional[str] = "",
    is_writing_tomtom_matrix: bool = False,
    top_n_matches: int = 3,
    trim_threshold: float = 0.3,
    trim_min_length: int = 3,
    tomtomlite: bool = False,
    num_workers: int = 4,
):
    output_dir = Path(f"{output_prefix}_report")
    output_dir.mkdir(parents=True, exist_ok=True)
    modiscolite.report.report_motifs(
        modisco_h5,
        output_dir.as_posix(),
        img_path_suffix,
        meme_motif_db=get_meme_file_path(meme_motif_db),
        is_writing_tomtom_matrix=is_writing_tomtom_matrix,
        top_n_matches=top_n_matches,
        trim_threshold=trim_threshold,
        trim_min_length=trim_min_length,
        ttl=tomtomlite,
        num_cores=num_workers,
        verbose=True,
    )


# all three function in one
def modisco(
    output_prefix: str,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[Union[str, int]] = 0,
    tss_distance: int = 1000,
    metadata_anndata: Optional[str] = None,
    genes: Optional[List[str]] = None,
    top_n_markers: Optional[int] = None,
    correct_grad: bool = True,
    num_workers: int = 4,
    genome: str = "hg38",
    method: str = "saliency",
    batch_size: int = 2,
    device: Optional[str] = None,
    # tfmodisco parameters
    sliding_window_size: int = 21,
    flank_size: int = 10,
    min_metacluster_size: int = 100,
    weak_threshold_for_counting_sign: float = 0.8,
    max_seqlets_per_metacluster: int = 20000,
    target_seqlet_fdr: float = 0.2,
    min_passing_windows_frac: float = 0.03,
    max_passing_windows_frac: float = 0.2,
    n_leiden_runs: int = 16,
    n_leiden_iterations: int = -1,
    min_overlap_while_sliding: float = 0.7,
    nearest_neighbors_to_compute: int = 500,
    affmat_correlation_threshold: float = 0.15,
    tsne_perplexity: float = 10.0,
    frac_support_to_trim_to: float = 0.2,
    min_num_to_trim_to: int = 30,
    trim_to_window_size: int = 30,
    initial_flank_to_add: int = 10,
    final_flank_to_add: int = 0,
    prob_and_pertrack_sim_merge_thresholds: List[Tuple[float, float]] = [(0.8, 0.8), (0.5, 0.85), (0.2, 0.9)],
    prob_and_pertrack_sim_dealbreaker_thresholds: List[Tuple[float, float]] = [
        (0.4, 0.75),
        (0.2, 0.8),
        (0.1, 0.85),
        (0.0, 0.9),
    ],
    subcluster_perplexity: int = 50,
    merging_max_seqlets_subsample: int = 300,
    final_min_cluster_size: int = 20,
    min_ic_in_window: float = 0.6,
    min_ic_windowsize: int = 6,
    ppm_pseudocount: float = 0.001,
    stranded: bool = False,
    pattern_type: str = "both",  # "both", "pos", or "neg"
    # reports parameters
    img_path_suffix: Optional[str] = "",
    meme_motif_db: Optional[Union[Path, str]] = "hocomoco_v13",
    is_writing_tomtom_matrix: bool = False,
    top_n_matches: int = 3,
    trim_threshold: float = 0.3,
    trim_min_length: int = 3,
    tomtomlite: bool = False,
):
    output_prefix = Path(output_prefix)

    if model == "ensemble":
        attrs_output_prefix = str(output_prefix) + "_{model}"
        models = [0, 1, 2, 3]
        attributions = [
            Path(attrs_output_prefix.format(model=model)).with_suffix(".attributions.h5") for model in models
        ]
    else:
        attrs_output_prefix = output_prefix
        models = [model]
        attributions = [output_prefix.with_suffix(".attributions.h5").as_posix()]

    for model in models:
        predict_save_modisco_attributions(
            output_prefix=str(attrs_output_prefix).format(model=model),
            tasks=tasks,
            off_tasks=off_tasks,
            model=model,
            metadata_anndata=metadata_anndata,
            genes=genes,
            top_n_markers=top_n_markers,
            method=method,
            batch_size=batch_size,
            device=device,
            num_workers=num_workers,
            genome=genome,
        )
    modisco_patterns(
        output_prefix=output_prefix,
        attributions=attributions,
        tasks=tasks,
        off_tasks=off_tasks,
        tss_distance=tss_distance,
        metadata_anndata=metadata_anndata,
        genes=genes,
        top_n_markers=top_n_markers,
        correct_grad=correct_grad,
        num_workers=num_workers,
        # tfmodisco parameters
        sliding_window_size=sliding_window_size,
        flank_size=flank_size,
        min_metacluster_size=min_metacluster_size,
        weak_threshold_for_counting_sign=weak_threshold_for_counting_sign,
        max_seqlets_per_metacluster=max_seqlets_per_metacluster,
        target_seqlet_fdr=target_seqlet_fdr,
        min_passing_windows_frac=min_passing_windows_frac,
        max_passing_windows_frac=max_passing_windows_frac,
        n_leiden_runs=n_leiden_runs,
        n_leiden_iterations=n_leiden_iterations,
        min_overlap_while_sliding=min_overlap_while_sliding,
        nearest_neighbors_to_compute=nearest_neighbors_to_compute,
        affmat_correlation_threshold=affmat_correlation_threshold,
        tsne_perplexity=tsne_perplexity,
        frac_support_to_trim_to=frac_support_to_trim_to,
        min_num_to_trim_to=min_num_to_trim_to,
        trim_to_window_size=trim_to_window_size,
        initial_flank_to_add=initial_flank_to_add,
        final_flank_to_add=final_flank_to_add,
        prob_and_pertrack_sim_merge_thresholds=prob_and_pertrack_sim_merge_thresholds,
        prob_and_pertrack_sim_dealbreaker_thresholds=prob_and_pertrack_sim_dealbreaker_thresholds,
        subcluster_perplexity=subcluster_perplexity,
        merging_max_seqlets_subsample=merging_max_seqlets_subsample,
        final_min_cluster_size=final_min_cluster_size,
        min_ic_in_window=min_ic_in_window,
        min_ic_windowsize=min_ic_windowsize,
        ppm_pseudocount=ppm_pseudocount,
        stranded=stranded,
        pattern_type=pattern_type,
    )
    modisco_reports(
        output_prefix=output_prefix,
        modisco_h5=output_prefix.with_suffix(".modisco.h5").as_posix(),
        img_path_suffix=img_path_suffix,
        meme_motif_db=meme_motif_db,
        is_writing_tomtom_matrix=is_writing_tomtom_matrix,
        top_n_matches=top_n_matches,
        trim_threshold=trim_threshold,
        trim_min_length=trim_min_length,
        tomtomlite=tomtomlite,
        num_workers=num_workers,
    )
