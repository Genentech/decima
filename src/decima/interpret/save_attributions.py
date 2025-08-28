import warnings
import logging
from pathlib import Path
from typing import List, Optional, Union

import h5py
from more_itertools import chunked
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyfaidx import Faidx
from grelu.sequence.format import convert_input_type

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.core.result import DecimaResult
from decima.data.dataset import GeneDataset, SeqDataset
from decima.hub import load_decima_model
from decima.interpret.attributer import DecimaAttributer
from decima.interpret.attributions import Attribution
from decima.utils import get_compute_device
from decima.utils.io import AttributionWriter, read_fasta_gene_mask
from decima.utils.task import _get_on_off_tasks, _get_genes
from decima.utils.qc import QCLogger


def _predict_save_attributions(
    output_prefix: str,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[int] = 0,
    metadata_anndata: Optional[str] = None,
    method: str = "inputxgradient",
    transform: str = "specificity",
    batch_size: int = 2,
    genes: Optional[List[str]] = None,
    seqs: Optional[Union[str, pd.DataFrame, np.ndarray, torch.Tensor]] = None,
    top_n_markers: Optional[int] = None,
    bigwig: bool = True,
    correct_grad_bigwig: bool = True,
    num_workers: int = 4,
    device: Optional[str] = None,
    genome: str = "hg38",
):
    warnings.filterwarnings("ignore", category=FutureWarning, module="tangermeme")

    logger = logging.getLogger("decima")

    device = get_compute_device(device)
    logger.info(f"Using device: {device}")

    logger.info("Loading model and metadata to compute attributions...")
    result = DecimaResult.load(metadata_anndata)

    tasks, off_tasks = _get_on_off_tasks(result, tasks, off_tasks)

    with QCLogger(output_prefix + "_attributions.qc.log", metadata_anndata=metadata_anndata) as qc:
        # TODO: qc.log_correlation(tasks, off_tasks)
        # TODO: qc.plot_correlation(tasks, off_tasks) save the plot to output_prefix + "qc/correlation.qc.png"

        attributer = DecimaAttributer.load_decima_attributer(model, tasks, off_tasks, method, transform, device=device)

        if (genes is not None) and (seqs is not None):
            raise ValueError("Only one of `genes` or `seqs` arguments must be provided not both.")
        elif seqs is not None:
            all_genes = [f"seq{i}" for i in range(len(seqs))]

            if isinstance(seqs, str):
                dataset = SeqDataset.from_fasta(seqs)
            elif isinstance(seqs, pd.DataFrame):
                dataset = SeqDataset.from_dataframe(seqs)
            elif isinstance(seqs, torch.Tensor) or isinstance(seqs, np.ndarray):
                assert seqs.shape[1] == 5, (
                    "`seqs` must be 5-dimensional with shape (batch_size, 5, seq_len) "
                    "where the 2th dimension is a one_hot encoded seq and binary mask gene mask."
                )
                dataset = TensorDataset(seqs)
                dataset.seqs = convert_input_type(seqs, output_type="strings")
            else:
                raise ValueError(f"Invalid type for seqs: {type(seqs)}. Must be a path to fasta file or pd.DataFrame.")
        else:
            all_genes = _get_genes(result, genes, top_n_markers, tasks, off_tasks)
            dataset = GeneDataset(genes=all_genes, metadata_anndata=result)
            genes = list(chunked(all_genes, batch_size))

        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )

        with AttributionWriter(
            path=Path(output_prefix).with_suffix(".attributions.h5"),
            genes=all_genes,
            model_name=attributer.model.name,
            metadata_anndata=result,
            genome=genome,
            bigwig=bigwig,
            correct_grad_bigwig=correct_grad_bigwig,
        ) as writer:
            for i, inputs in enumerate(tqdm(dl, desc="Computing attributions...")):
                attrs = attributer.attribute(inputs.to(device)).detach().cpu().numpy()
                seqs = inputs[:, :4].detach().cpu().numpy()

                for gene, attr, seq in zip(genes[i], attrs, seqs):
                    writer.add(gene=gene, seqs=seq, attrs=attr)
                    qc.log_gene(gene, threshold=0.7)

        if seqs is not None:
            logger.info("Saving sequences...")
            fasta_path = str(output_prefix + ".seqs.fasta")
            with open(fasta_path, "w") as f:
                for i, seq in tqdm(zip(all_genes, dataset.seqs), desc="Saving sequences..."):
                    f.write(f">{i}\n{seq}\n")
            Faidx(fasta_path, build_index=True)


def recursive_seqlet_calling(
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
):
    raise NotImplementedError("Recursive seqlet calling is not implemented yet.")


def predict_save_attributions(
    output_dir: str,
    genes: Optional[Union[str, List[str]]] = None,
    seqs: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[Union[str, int]] = 0,
    metadata_anndata: Optional[str] = None,
    method: str = "inputxgradient",
    device: Optional[str] = None,
    plot_peaks: bool = True,
    plot_seqlogo: bool = False,
    seqlogo_window: int = 50,
    dpi: int = 100,
):
    """Generate and save attribution analysis results for a gene.
    This function performs attribution analysis for a given gene and cell types, saving
    the following output files to the specified directory:

    output_dir/
    ├── peaks.bed                # List of attribution peaks in BED format
    ├── peaks.png                # Plot showing peak locations
    ├── qc.log                   # QC warnings about prediction reliability
    ├── motifs.tsv               # Detected motifs in peak regions
    ├── attributions.h5          # Raw attribution score matrix
    ├── attributions.bigwig      # Genome browser track of attribution scores
    └── attributions_seq_logos/  # Directory containing attribution plots
        └── {peak}.png           # Attribution plot for each peak region

    Args:
        output_dir: Directory to save output files
        gene: Gene symbol or ID to analyze
        tasks: List of cell types to analyze attributions for
        off_tasks: Optional list of cell types to contrast against
        model: Optional model to use for attribution analysis
        method: Method to use for attribution analysis
        device: Device to use for attribution analysis
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
    warnings.filterwarnings("ignore", category=FutureWarning, module="tangermeme")

    if (genes is None) and (seqs is None):
        raise ValueError("Either genes or seq must be provided")

    if (genes is not None) and (seqs is not None):
        raise ValueError("Only one of genes or seq must be provided")

    logger = logging.getLogger("decima")

    device = get_compute_device(device)
    logger.info(f"Using device: {device}")

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"Output directory {output_dir} already exists. Please delete it or use a different directory."
        )

    logger.info("Loading model and metadata to compute attributions...")
    result = DecimaResult.load(metadata_anndata)

    if tasks is None:
        tasks = result.cell_metadata.index.tolist()
    elif isinstance(tasks, str):
        tasks = result.query_cells(tasks)

    if isinstance(off_tasks, str):
        off_tasks = result.query_cells(off_tasks)

    attributions = list()

    if genes is not None:
        result.load_model(model=model, device=device)

        if isinstance(genes, str):
            genes = genes.split(",")

        with QCLogger(output_dir / "qc.warnings.log", metadata_anndata=metadata_anndata) as qc:
            for gene in genes:
                qc.log_gene(gene, threshold=0.7)

        for gene in tqdm(genes, desc="Computing attributions..."):
            attributions.append(result.attributions(gene, tasks, off_tasks, method=method))

    else:
        if isinstance(seqs, str):
            seqs = list(read_fasta_gene_mask(seqs).itertuples())
        elif isinstance(seqs, pd.DataFrame):
            assert (
                ("seq" in seqs.columns) and ("gene_mask_start" in seqs.columns) and ("gene_mask_end" in seqs.columns)
            ), "`seqs` must contain `seq`, `gene_mask_start`, and `gene_mask_end` columns."
            seqs = list(seqs.itertuples())
        elif isinstance(seqs, torch.Tensor) or isinstance(seqs, np.ndarray):
            assert seqs.shape[1] == 5, (
                "`seqs` must be 5-dimensional with shape (batch_size, 5, seq_len) "
                "where the 2th dimension is a one_hot encoded seq and binary mask gene mask."
            )
        else:
            raise ValueError(
                f"Invalid type for seqs: {type(seqs)}. Must be a path to fasta file, pd.DataFrame, or torch.Tensor or np.array."
            )

        model = load_decima_model(model, device=device)

        for i, row in tqdm(enumerate(seqs), desc="Computing attributions..."):
            if isinstance(row, tuple):
                seq = row.seq
                gene = str(row.Index)
                gene_mask_start = row.gene_mask_start
                gene_mask_end = row.gene_mask_end
            else:
                seq = row
                gene = str(i)
                gene_mask_start = None
                gene_mask_end = None

            attributions.append(
                Attribution.from_seq(
                    inputs=seq,
                    tasks=tasks,
                    off_tasks=off_tasks,
                    model=model,
                    method=method,
                    device=device,
                    gene=gene,
                    gene_mask_start=gene_mask_start,
                    gene_mask_end=gene_mask_end,
                )
            )

    logger.info("Saving attribution scores...")
    with h5py.File(output_dir / "attributions.h5", "w") as f:
        for attrs in tqdm(attributions, desc="Saving attributions..."):
            f.create_dataset(attrs.gene, data=attrs.attrs, chunks=(4, DECIMA_CONTEXT_SIZE), compression="gzip")

    logger.info("Saving peaks...")
    df_peaks = pd.concat([attrs.peaks_to_bed() for attrs in tqdm(attributions, desc="Saving peaks...")])
    df_peaks.to_csv(output_dir / "peaks.bed", sep="\t", header=False, index=False)

    if plot_peaks:
        logger.info("Saving peaks plots...")
        plt.ioff()
        peak_plot_dir = output_dir / "peaks_plots"
        peak_plot_dir.mkdir()
        for attrs in tqdm(attributions, desc="Saving peaks plots..."):
            attrs.plot_peaks().save(peak_plot_dir / f"{attrs.gene}.png", dpi=dpi, verbose=False)
        plt.close()

    logger.info("Scanning for motifs...")
    df_motifs = pd.concat([attrs.scan_motifs() for attrs in tqdm(attributions, desc="Scanning for motifs...")])
    df_motifs.to_csv(output_dir / "motifs.tsv", sep="\t", index=False)

    logger.info("Saving coverage...")
    cov_dir = output_dir / "coverage"
    cov_dir.mkdir(exist_ok=True)
    for attrs in tqdm(attributions, desc="Saving coverage..."):
        attrs.save_bigwig(str(cov_dir / f"{attrs.gene}.bw"))

    if plot_seqlogo:
        logger.info("Generating attribution plots...")
        seqlogo_plot_dir = output_dir / "seqlogos"
        seqlogo_plot_dir.mkdir()

        plt.ioff()
        for attrs in tqdm(attributions, desc="Generating attribution plots..."):
            for peak in attrs.peaks.itertuples():
                logo = attrs.plot_seqlogo(relative_loc=peak.from_tss, window=seqlogo_window)
                logo.ax.figure.savefig(seqlogo_plot_dir / f"{attrs.gene}@{peak.from_tss}.png")

    if seqs is not None:
        logger.info("Saving sequences...")
        fasta_path = str(output_dir / "seqs.fasta")
        with open(fasta_path, "w") as f:
            for attrs in tqdm(attributions, desc="Saving sequences..."):
                f.write(attrs.fasta_str())
        Faidx(fasta_path, build_index=True)
