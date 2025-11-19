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
import fastermodiscolite
from tqdm import tqdm
from grelu.resources import get_meme_file_path
from grelu.interpret.motifs import trim_pwm

from decima.constants import DECIMA_CONTEXT_SIZE, DEFAULT_ENSEMBLE
from decima.core.result import DecimaResult
from decima.utils import _get_on_off_tasks, _get_genes
from decima.utils.motifs import motif_start_end
from decima.core.attribution import AttributionResult
from decima.interpret.attributions import predict_save_attributions


def predict_save_modisco_attributions(
    output_prefix: str,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[Union[str, int]] = DEFAULT_ENSEMBLE,
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
    """Generate and save attribution analysis results optimized for MoDISco motif discovery.

    This function performs attribution analysis for specified genes and cell types, generating
    attribution scores that will be used downstream for MoDISco pattern discovery and motif analysis.

    Args:
        output_prefix: Prefix for the output files where attribution results will be saved.
        tasks: Tasks to analyze for modisco attribution either list of task names or query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte''). If not provided, all tasks will be analyzed.
        off_tasks: Off tasks to analyze for modisco attribution either list of task names or query string to filter cell types to contrast against (e.g. 'cell_type == 'classical monocyte''). If not provided, no contrast will be performed.
        model: Model to use for attribution analysis default is 0. Can be replicate number (0-3) or path to custom model.
        metadata_anndata: Metadata anndata path or DecimaResult object. If not provided, the default metadata will be downloaded from wandb.
        method: Method to use for attribution analysis default is "saliency". Available options: "saliency", "inputxgradient", "integratedgradients". For MoDISco, "saliency" is often preferred for pattern discovery.
        transform: Transform to use for attribution analysis default is "specificity". Available options: "specificity", "aggregate". Specificity transform is recommended for MoDISco to highlight cell-type-specific patterns.
        batch_size: Batch size for attribution analysis default is 1. Increasing batch size may speed up computation but requires more memory.
        genes: Genes to analyze for modisco attribution if not provided, all genes will be used. Can be list of gene symbols or IDs to focus analysis on specific genes.
        top_n_markers: Top n markers for modisco attribution if not provided, all markers will be analyzed. Useful for focusing on the most important marker genes for the specified tasks.
        bigwig: Whether to save attribution scores as a bigwig file default is True. Bigwig files can be loaded in genome browsers for visualization.
        correct_grad_bigwig: Whether to correct the gradient bigwig file default is True. Applies gradient correction for better visualization quality.
        num_workers: Number of workers for attribution analysis default is 4. Increasing number of workers will speed up the process but requires more memory.
        device: Device to use for attribution analysis (e.g. 'cuda', 'cpu'). If not provided, the best available device will be used automatically.
        genome: Genome to use for attribution analysis default is "hg38". Can be genome name or path to custom genome fasta file.

    Examples:
    >>> predict_save_modisco_attributions(
    ...     output_dir="output_dir",
    ...     tasks="cell_type == 'classical monocyte'",
    ... )
    """
    return predict_save_attributions(
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
    """Perform TF-MoDISco pattern discovery and motif clustering from attribution data.

    This function runs the core TF-MoDISco algorithm to discover recurring patterns (motifs)
    in attribution data by clustering similar seqlets and identifying consensus motifs.

    Args:
        output_prefix: Prefix for the output files where MoDISco results will be saved. Results will be saved as "{output_prefix}.modisco.h5".
        attributions: Path to attribution file(s) or list of attribution files containing computed attribution scores from previous analysis.
        tasks: Tasks to analyze either list of task names or query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte''). If not provided, all tasks will be analyzed.
        off_tasks: Off tasks to analyze either list of task names or query string to filter cell types to contrast against (e.g. 'cell_type == 'classical monocyte''). If not provided, all tasks will be used as off tasks.
        tss_distance: Distance from TSS to analyze for pattern discovery default is 10000. Controls the genomic window size around TSS for seqlet detection and motif discovery.
        metadata_anndata: Name of the model or path to metadata anndata file or DecimaResult object. If not provided, the compatible metadata of the saved attribution files will be used.
        genes: Genes to analyze for pattern discovery if not provided, all genes will be used. Can be list of gene symbols or IDs to focus analysis on specific genes.
        top_n_markers: Top n markers to analyze for pattern discovery if not provided, all markers will be analyzed. Useful for focusing on the most important marker genes for the specified tasks.
        correct_grad: Whether to correct gradient for attribution analysis default is True. Applies gradient correction for better attribution quality before pattern discovery.
        num_workers: Number of workers for parallel processing default is 4. Increasing number of workers will speed up the process but requires more memory.
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

    if isinstance(attributions, (str, Path)):
        attributions_files = [Path(attributions).as_posix()]
    else:
        attributions_files = attributions

    with AttributionResult(
        attributions_files, tss_distance, correct_grad, num_workers=num_workers, agg_func="mean"
    ) as ar:
        genome = ar.genome
        model_names = ar.model_name

        metadata_anndata = metadata_anndata or model_names[0]
        logger.info(f"Loading metadata for model {metadata_anndata}...")
        result = DecimaResult.load(metadata_anndata)

        tasks, off_tasks = _get_on_off_tasks(result, tasks, off_tasks)
        all_genes = _get_genes(result, genes, top_n_markers, tasks, off_tasks)
        sequences, attributions = ar.load(all_genes)

    pos_patterns, neg_patterns = fastermodiscolite.tfmodisco.TFMoDISco(
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
    fastermodiscolite.io.save_hdf5(
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
    """Generate comprehensive HTML reports and motif comparisons from MoDISco results.

    This function takes MoDISco pattern discovery results and generates detailed HTML reports
    including motif visualizations, database comparisons, and statistical summaries.

    Args:
        output_prefix: Prefix for the output report files where results will be saved. A "_report" suffix will be added to create the output directory.
        modisco_h5: Path to the MoDISco HDF5 file containing discovered patterns and motifs from previous MoDISco analysis.
        meme_motif_db: MEME motif database for comparison default is "hocomoco_v13". Database used for motif comparison and annotation. Can be database name or path to custom MEME format database.
        img_path_suffix: Image path suffix for output plots default is "". Optional suffix to add to image file paths for organizational purposes.
        is_writing_tomtom_matrix: Whether to write TOMTOM comparison matrix default is False. If True, outputs detailed comparison matrix between discovered and database motifs for downstream analysis.
        top_n_matches: Top n matches to report default is 3. Number of top database matches to report for each discovered motif in the HTML output.
        trim_threshold: Trim threshold for motif boundaries default is 0.3. Threshold for determining where to trim motif boundaries based on information content when generating logos.
        trim_min_length: Minimum trim length default is 3. Minimum number of positions to retain when trimming motifs to ensure meaningful motif representations.
        tomtomlite: Whether to use TOMTOM lite mode default is False. If True, uses a faster but less comprehensive version of TOMTOM for motif comparison.
        num_workers: Number of workers for parallel processing default is 4. Increasing number of workers will speed up report generation but requires more memory.

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
    fastermodiscolite.report.report_motifs(
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
    metadata_anndata: Optional[str] = None,
    trim_threshold: float = 0.2,
):
    """Extract seqlet locations from MoDISco results and save as BED format file.

    This function processes MoDISco pattern discovery results to extract the genomic coordinates
    of discovered seqlets (sequence motifs) and outputs them in standard BED format for
    downstream analysis and visualization in genome browsers.

    Args:
        output_prefix: Prefix for the output BED file where seqlet coordinates will be saved. The output will be saved as "{output_prefix}.seqlets.bed".
        modisco_h5: Path to the MoDISco HDF5 file containing discovered patterns and seqlet information from previous MoDISco analysis.
        metadata_anndata: Path to metadata anndata file or DecimaResult object default is None. Required for mapping seqlet coordinates to genomic positions. If not provided, relative coordinates will be used.
        trim_threshold: Trim threshold for seqlet boundaries default is 0.2. Threshold for determining seqlet boundaries based on contribution scores - lower values result in longer seqlets.

    Examples:
        >>> modisco_seqlet_bed(
        ...     output_prefix="my_analysis",
        ...     modisco_h5="my_analysis.modisco.h5",
        ...     metadata_anndata="metadata.h5ad",
        ...     trim_threshold=0.15,
        ... )
    """

    df = list()

    with h5py.File(modisco_h5, "r") as f:
        model_name = f.attrs["model_names"].split(",")[0]
        result = DecimaResult.load(metadata_anndata or model_name)

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
    model: Optional[Union[str, int]] = DEFAULT_ENSEMBLE,
    tss_distance: int = 1000,
    metadata_anndata: Optional[str] = None,
    genes: Optional[List[str]] = None,
    top_n_markers: Optional[int] = None,
    correct_grad: bool = True,
    num_workers: int = 4,
    genome: str = "hg38",
    method: str = "saliency",
    transform: Optional[str] = "specificity",
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
        output_prefix: Path prefix to save comprehensive modisco results where all output files will be written.
        tasks: Tasks to analyze for full modisco pipeline either list of task names or query string to filter cell types to analyze attributions for (e.g. 'cell_type == 'classical monocyte''). If not provided, all tasks will be analyzed.
        off_tasks: Off tasks to analyze for full modisco pipeline either list of task names or query string to filter cell types to contrast against (e.g. 'cell_type == 'classical monocyte''). If not provided, no contrast will be performed.
        model: Model to use for attribution analysis default is 0. Can be replicate number (0-3) or path to custom model.
        tss_distance: Distance from TSS to call seqlets default is 1000. Controls the genomic window size around TSS for seqlet detection. If set to full context size of decima (524288), analyzes the entire accessible region.
        metadata_anndata: Path to metadata anndata file or DecimaResult object. If not provided, the default metadata will be downloaded from wandb.
        genes: List of genes to analyze for full modisco pipeline if not provided, all genes will be used. Can be list of gene symbols or IDs to focus analysis on specific genes.
        top_n_markers: Top n markers to analyze for full modisco pipeline if not provided, all markers will be analyzed. Useful for focusing on the most important marker genes for the specified tasks.
        correct_grad: Whether to correct gradient for attribution analysis default is True. Applies gradient correction for better attribution quality.
        num_workers: Number of workers for parallel processing default is 4. Increasing number of workers will speed up the process but requires more memory.
        genome: Genome reference to use default is "hg38". Can be genome name or path to custom genome fasta file.
        method: Method to use for attribution analysis default is "saliency". Available options: "saliency", "inputxgradient", "integratedgradients". For MoDISco, "saliency" is often preferred for pattern discovery.
        transform: Transform to use for attribution analysis default is "specificity". Available options: "specificity", "aggregate". Specificity transform is recommended for MoDISco to highlight cell-type-specific patterns.
        batch_size: Batch size for attribution analysis default is 2. Increasing batch size may speed up computation but requires more memory.
        device: Device to use for computation (e.g. 'cuda', 'cpu'). If not provided, the best available device will be used automatically.
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
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    attributions_paths = predict_save_modisco_attributions(
        output_prefix=output_prefix,
        tasks=tasks,
        off_tasks=off_tasks,
        model=model,
        metadata_anndata=metadata_anndata,
        genes=genes,
        top_n_markers=top_n_markers,
        method=method,
        transform=transform,
        batch_size=batch_size,
        correct_grad_bigwig=correct_grad,
        device=device,
        num_workers=num_workers,
        genome=genome,
    )
    modisco_patterns(
        output_prefix=output_prefix,
        attributions=attributions_paths,
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
