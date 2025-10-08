"""Modisco module perform modisco motif clustering from attributions.

Examples:
    >>> predict_save_modisco_attributions(
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
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import modiscolite
from tqdm import tqdm
from grelu.resources import get_meme_file_path
from grelu.interpret.motifs import trim_pwm

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.core.result import DecimaResult
from decima.utils import _get_on_off_tasks, _get_genes
from decima.utils.motifs import motif_start_end
from decima.core.attribution import AttributionResult
from decima.interpret.attributions import predict_save_attributions


def predict_save_modisco_attributions(
    output_prefix: str,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[int] = 0,
    metadata_anndata: Optional[str] = None,
    method: str = "saliency",
    transform: str = "specificity",
    batch_size: int = 1,
    genes: Optional[List[str]] = None,
    top_n_markers: Optional[int] = None,
    bigwig: bool = True,
    correct_grad_bigwig: bool = True,
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
        bigwig: Whether to save bigwig file.
        correct_grad_bigwig: Whether to correct gradient for bigwig file.
        num_workers: Number of workers for the prediction.
        device: Device to use for attribution analysis.
        genome: Genome name or path to the genome fasta file.

    Raises:
        FileExistsError: If output directory already exists.

    Examples:
    >>> predict_save_modisco_attributions(
    ...     output_dir="output_dir",
    ...     tasks="cell_type == 'classical monocyte'",
    ... )
    """
    predict_save_attributions(
        output_prefix=output_prefix,
        tasks=tasks,
        off_tasks=off_tasks,
        model=model,
        metadata_anndata=metadata_anndata,
        method=method,
        transform=transform,
        batch_size=batch_size,
        genes=genes,
        top_n_markers=top_n_markers,
        bigwig=bigwig,
        correct_grad_bigwig=correct_grad_bigwig,
        num_workers=num_workers,
        device=device,
        genome=genome,
    )


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
    """Perform modisco motif clustering from attributions.

    Args:
        output_prefix: Path to save modisco results.
        attributions: Path to attributions file.
        tasks: List of tasks to analyze.
        off_tasks: List of off tasks to analyze.
        tss_distance: TSS distance.
        metadata_anndata: Path to metadata anndata file.
        genes: List of genes to analyze.
        top_n_markers: Top n markers to analyze.
        correct_grad: Whether to correct gradient.
        num_workers: Number of workers.
        sliding_window_size: Sliding window size.
        flank_size: Flank size.
        min_metacluster_size: Min metacluster size.
        weak_threshold_for_counting_sign: Weak threshold for counting sign.
        max_seqlets_per_metacluster: Max seqlets per metacluster.
        target_seqlet_fdr: Target seqlet FDR.
        min_passing_windows_frac: Min passing windows fraction.
        max_passing_windows_frac: Max passing windows fraction.
        n_leiden_runs: Number of Leiden runs.
        n_leiden_iterations: Number of Leiden iterations.
        min_overlap_while_sliding: Min overlap while sliding.
        nearest_neighbors_to_compute: Nearest neighbors to compute.
        affmat_correlation_threshold: Affmat correlation threshold.
        tsne_perplexity: TSNE perplexity.
        frac_support_to_trim_to: Frac support to trim to.
        min_num_to_trim_to: Min num to trim to.
        trim_to_window_size: Trim to window size.
        initial_flank_to_add: Initial flank to add.
        final_flank_to_add: Final flank to add.
        prob_and_pertrack_sim_merge_thresholds: Prob and pertrack sim merge thresholds.
        prob_and_pertrack_sim_dealbreaker_thresholds: Prob and pertrack sim dealbreaker thresholds.
        subcluster_perplexity: Subcluster perplexity.
        merging_max_seqlets_subsample: Merging max seqlets subsample.
        final_min_cluster_size: Final min cluster size.
        min_ic_in_window: Min IC in window.
        min_ic_windowsize: Min IC windowsize.
        ppm_pseudocount: PPM pseudocount.
        stranded: Stranded.
        pattern_type: Pattern type.

    Examples:
        >>> modisco_patterns(
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
    logger = logging.getLogger("decima")
    logger.info("Loading metadata")
    result = DecimaResult.load(metadata_anndata)

    if isinstance(attributions, (str, Path)):
        attributions_files = [Path(attributions).as_posix()]
    else:
        attributions_files = attributions

    tasks, off_tasks = _get_on_off_tasks(result, tasks, off_tasks)
    all_genes = _get_genes(result, genes, top_n_markers, tasks, off_tasks)

    with AttributionResult(attributions_files, tss_distance, correct_grad, num_workers=1, agg_func="mean") as ar:
        sequences, attributions = ar.load(all_genes)
        genome = ar.genome
        model_names = ar.model_name

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
    h5_path = Path(output_prefix).with_suffix(".modisco.h5").as_posix()
    modiscolite.io.save_hdf5(
        h5_path,
        pos_patterns,
        neg_patterns,
        window_size=tss_distance * 2 if tss_distance is not None else DECIMA_CONTEXT_SIZE,
    )
    with h5py.File(h5_path, "a") as f:
        f.create_dataset("genes", data=np.array(all_genes, dtype="S100"))
        f.attrs["tss_distance"] = tss_distance
        f.attrs["model_names"] = ",".join(model_names)
        f.attrs["genome"] = genome


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
    """Perform modisco motif clustering from attributions.

    Args:
        output_prefix: Path to save modisco results.
        modisco_h5: Path to modisco h5 file.
        meme_motif_db: Path to meme motif db.
        img_path_suffix: Image path suffix.
        is_writing_tomtom_matrix: Whether to write tomtom matrix.
        top_n_matches: Top n matches.
        trim_threshold: Trim threshold.
        trim_min_length: Trim min length.
        tomtomlite: Whether to use tomtomlite.
        num_workers: Number of workers.

    Examples:
        >>> modisco_reports(
        ...     output_prefix="output_prefix",
        ...     modisco_h5="modisco.h5",
        ...     meme_motif_db="hocomoco_v13",
        ...     img_path_suffix="",
        ... )
    """
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


def modisco_seqlet_bed(
    output_prefix: str,
    modisco_h5: str,
    metadata_anndata: str = None,
    trim_threshold: float = 0.2,
):
    """Perform modisco seqlet bed from attributions.

    Args:
        output_prefix: Path to save modisco results.
        modisco_h5: Path to modisco h5 file.
        metadata_anndata: Path to metadata anndata file.
        trim_threshold: Trim threshold.
    """
    result = DecimaResult.load(metadata_anndata)

    df = list()

    with h5py.File(modisco_h5, "r") as f:
        tss_distance = f.attrs["tss_distance"]
        genes = [gene.decode("utf-8") for gene in f["genes"][:]]
        genes_idx = dict(enumerate(genes))
        df_genes = result.gene_metadata.loc[genes]

        for pattern_type in ["pos_patterns", "neg_patterns"]:
            if pattern_type in f:
                for pattern_name in tqdm(f[pattern_type].keys(), desc=f"Processing {pattern_type} patterns..."):
                    pattern = f[pattern_type][pattern_name]
                    pattern_seqlets = pattern["seqlets"]

                    _genes = [genes_idx[idx] for idx in pattern_seqlets["example_idx"][:]]

                    cwm_seqlet = pattern_seqlets["contrib_scores"][:]
                    motif_starts, motif_ends = motif_start_end(
                        cwm_seqlet,
                        trim_pwm(pattern["contrib_scores"][:].T, trim_threshold=trim_threshold).T,
                    )
                    df.append(
                        pd.DataFrame(
                            {
                                "_start": pattern_seqlets["start"][:].tolist(),
                                "_end": pattern_seqlets["end"][:].tolist(),
                                "name": [
                                    f"{pattern_type}.{pattern_name}.seqlet_{i}.{gene}" for i, gene in enumerate(_genes)
                                ],
                                "score": cwm_seqlet.sum((1, 2)).tolist(),
                                "revcomp": pattern_seqlets["is_revcomp"][:].tolist(),
                                "gene": _genes,
                                "_motif_start": motif_starts,
                                "_motif_end": motif_ends,
                            }
                        )
                    )

    df = pd.concat(df).set_index("gene").join(df_genes[["chrom", "strand", "gene_start", "gene_end"]], on="gene")

    # `_strand` of the seqlet, `strand` of the gene, `revcomp` of the motif respect to the gene
    df["_strand"] = np.where(df["revcomp"], df["strand"].replace({"+": "-", "-": "+"}), df["strand"])
    tss_pos = np.where(df["strand"] == "+", df["gene_start"], df["gene_end"])

    start = np.where(df["strand"] == "+", df["_start"], 2 * tss_distance - df["_end"])
    end = np.where(df["strand"] == "+", df["_end"], 2 * tss_distance - df["_start"])

    lengths = df["_end"] - df["_start"]
    motif_start = np.where(df["revcomp"], lengths - df["_motif_end"], df["_motif_start"])
    motif_end = np.where(df["revcomp"], lengths - df["_motif_start"], df["_motif_end"])

    region_start = tss_pos - tss_distance
    df["start"] = region_start + start
    df["end"] = region_start + end
    df["motif_start"] = df["start"] + motif_start
    df["motif_end"] = df["start"] + motif_end + 1
    df["color"] = "65,105,225"  # blue # TO THINK: can be colored by significance

    df[["chrom", "start", "end", "name", "score", "_strand", "motif_start", "motif_end", "color"]].sort_values(
        ["chrom", "start"]
    ).to_csv(Path(output_prefix).with_suffix(".seqlets.bed"), sep="\t", header=False, index=False)


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
    # seqlet thresholds
    seqlet_motif_trim_threshold: float = 0.2,
):
    """Perform modisco motif clustering from attributions.

    Args:
        output_prefix: Path to save modisco results.
        tasks: List of tasks to analyze.
        off_tasks: List of off tasks to analyze.
        model: Model to analyze.
        tss_distance: TSS distance.
        metadata_anndata: Path to metadata anndata file.
        genes: List of genes to analyze.
        top_n_markers: Top n markers to analyze.
        correct_grad: Whether to correct gradient.
        num_workers: Number of workers.
        genome: Genome.
        method: Method to analyze.
        batch_size: Batch size.
        device: Device to analyze.
        sliding_window_size: Sliding window size.
        flank_size: Flank size.
        min_metacluster_size: Min metacluster size.
        weak_threshold_for_counting_sign: Weak threshold for counting sign.
        max_seqlets_per_metacluster: Max seqlets per metacluster.
        target_seqlet_fdr: Target seqlet FDR.
        min_passing_windows_frac: Min passing windows fraction.
        max_passing_windows_frac: Max passing windows fraction.
        n_leiden_runs: Number of Leiden runs.
        n_leiden_iterations: Number of Leiden iterations.
        min_overlap_while_sliding: Min overlap while sliding.
        nearest_neighbors_to_compute: Nearest neighbors to compute.
        affmat_correlation_threshold: Affmat correlation threshold.
        tsne_perplexity: TSNE perplexity.
        frac_support_to_trim_to: Frac support to trim to.
        min_num_to_trim_to: Min num to trim to.
        trim_to_window_size: Trim to window size.
        initial_flank_to_add: Initial flank to add.
        final_flank_to_add: Final flank to add.
        prob_and_pertrack_sim_merge_thresholds: Prob and pertrack sim merge thresholds.
        prob_and_pertrack_sim_dealbreaker_thresholds: Prob and pertrack sim dealbreaker thresholds.
        subcluster_perplexity: Subcluster perplexity.
        merging_max_seqlets_subsample: Merging max seqlets subsample.
        final_min_cluster_size: Final min cluster size.
        min_ic_in_window: Min IC in window.
        min_ic_windowsize: Min IC windowsize.
        ppm_pseudocount: PPM pseudocount.
        stranded: Stranded.
        pattern_type: Pattern type.
        img_path_suffix: Image path suffix.
        meme_motif_db: Meme motif db.
        is_writing_tomtom_matrix: Whether to write tomtom matrix.
        top_n_matches: Top n matches.
        trim_threshold: Trim threshold.
        trim_min_length: Trim min length.
        tomtomlite: Whether to use tomtomlite.
        seqlet_motif_trim_threshold: Seqlet motif trim threshold.
    """
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
            correct_grad_bigwig=correct_grad,
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
    modisco_seqlet_bed(
        output_prefix=output_prefix,
        modisco_h5=output_prefix.with_suffix(".modisco.h5").as_posix(),
        metadata_anndata=metadata_anndata,
        trim_threshold=seqlet_motif_trim_threshold,
    )
