import logging
from pathlib import Path
from typing import List, Optional, Union
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from decima.core.result import DecimaResult
from decima.interpret.attribution import Attribution
from decima.utils import get_compute_device


def predict_save_attributions(
    output_dir: str,
    genes: Optional[Union[str, List[str]]] = None,
    seqs: Optional[Union[pd.DataFrame, np.ndarray, torch.Tensor]] = None,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
    model: Optional[Union[str, int]] = 0,
    device: Optional[str] = None,
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
        device: Device to use for attribution analysis
        dpi: DPI for attribution plots.

    Raises:
        FileExistsError: If output directory already exists.

    Examples:
    >>> predict_save_attributions(
    ...     output_dir="output_dir",
    ...     gene="SPI1",
    ...     tasks="cell_type == 'classical monocyte'",
    ... )
    """

    if genes is None and seqs is None:
        raise ValueError("Either genes or seq must be provided")

    if genes is not None and seqs is not None:
        raise ValueError("Only one of genes or seq must be provided")

    logger = logging.getLogger(__name__)

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

    if genes is not None:
        genes = genes.split(",")
        gene = genes[0]
        result = DecimaResult.load()
        result.load_model(model=model, device=device)
        attributions = result.attributions(gene, tasks, off_tasks)
    else:
        row = seqs.iloc[0]
        attributions = Attribution.from_seq(
            seqs,
            tasks,
            off_tasks,
            model=model,
            device=device,
            gene=row.name,
            gene_mask_start=row.gene_mask_start,
            gene_mask_end=row.gene_mask_end,
        )

    logger.info("Writing QC log...")
    with open(output_dir / "qc.log", "w") as f:
        f.write("TODO: write qc log based on the gene correlation")

    logger.info("Saving peaks...")
    df_peaks = attributions.peaks_to_bed()
    df_peaks.to_csv(output_dir / "peaks.bed", sep="\t", header=False, index=False)

    plt.ioff()
    attributions.plot_peaks().save(output_dir / f"{attributions.gene}.png", dpi=dpi)
    plt.close()

    logger.info("Scanning for motifs...")
    attributions.scan_motifs().to_csv(output_dir / "motifs.tsv", sep="\t", index=False)

    logger.info("Saving attribution scores...")

    with h5py.File(output_dir / "attributions.h5", "w") as f:
        f.create_dataset(attributions.gene, data=attributions.attrs)

    cov_dir = output_dir / "coverage"
    cov_dir.mkdir(exist_ok=True)
    attributions.save_bigwig(str(cov_dir / f"{attributions.gene}.bw"))

    if plot_seqlogo:
        logger.info("Generating attribution plots...")
        peak_plot_dir = output_dir / "seqlogos"
        peak_plot_dir.mkdir()

        plt.ioff()
        for peak in attributions.peaks.itertuples():
            logo = attributions.plot_seqlogo(relative_loc=peak.from_tss, window=seqlogo_window)
            logo.ax.figure.savefig(peak_plot_dir / f"{attributions.gene}@{peak.from_tss}.png")

    if seqs is not None:
        attributions.save_fasta(str(output_dir / "seqs.fasta"))
