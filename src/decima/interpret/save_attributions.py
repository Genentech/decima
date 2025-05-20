import logging
from typing import List, Optional, Union
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from decima.core.result import DecimaResult


def predict_save_attributions(
    output_dir: str,
    gene: str,
    tasks: List[str],
    off_tasks: Optional[List[str]] = None,
    model: Optional[Union[str, int]] = 0,
    device: str = "cpu",
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
    ├── attributions.npz         # Raw attribution score matrix
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
    """
    logger = logging.getLogger(__name__)

    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(
            f"Output directory {output_dir} already exists. Please delete it or use a different directory."
        )

    logger.info("Loading model and metadata to compute attributions...")

    result = DecimaResult.load()
    result.load_model(model=model, device=device)
    attributions = result.attributions(gene, tasks, off_tasks)

    logger.info("Writing QC log...")
    with open(output_dir / "qc.log", "w") as f:
        f.write("TODO: write qc log based on the gene correlation")

    # save info as yaml

    logger.info("Saving peaks...")
    df_peaks = attributions.peaks_to_bed()
    df_peaks.to_csv(output_dir / "peaks.bed", sep="\t", header=False, index=False)

    plt.ioff()
    attributions.plot_peaks().save(output_dir / "peaks.png", dpi=dpi)
    plt.close()

    logger.info("Scanning for motifs...")
    attributions.scan_motifs().to_csv(output_dir / "motifs.tsv", sep="\t", index=False)

    logger.info("Saving attribution scores...")
    np.savez_compressed(output_dir / "attributions.npz", attributions=attributions.attrs)
    attributions.save_bigwig(str(output_dir / "attributions.bigwig"))

    logger.info("Generating attribution plots...")
    peak_plot_dir = output_dir / "attributions_seq_logos"
    peak_plot_dir.mkdir()

    plt.ioff()
    for peak in attributions.peaks.itertuples():
        logo = attributions.plot_attributions(relative_loc=peak.from_tss, window=50)  # TODO: window to args
        logo.ax.figure.savefig(peak_plot_dir / f"{attributions.gene}@{peak.from_tss}.png")
