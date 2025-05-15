from typing import List
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyBigWig
import genomepy

from decima.core.result import DecimaResult


def predict_save_attributions(output_dir: str, gene: str, cells: List[str], constract_cells: List[str] = None):
    """Generate and save attribution analysis results for a gene.
    This function performs attribution analysis for a given gene and cell types, saving
    the following output files to the specified directory:
    
    output_dir/
    ├── peaks.bed           # List of attribution peaks in BED format
    ├── peaks.png          # Plot showing peak locations 
    ├── qc.log            # QC warnings about prediction reliability
    ├── motifs.tsv        # Detected motifs in peak regions
    ├── attributions_scores.tsv  # Raw attribution score matrix
    ├── attributions.bigwig      # Genome browser track of attribution scores
    └── attributions/           # Directory containing attribution plots
        └── {peak}.png         # Attribution plot for each peak region

    Args:
        output_dir: Directory to save output files
        gene: Gene symbol or ID to analyze
        cells: List of cell types to analyze attributions for
        constract_cells: Optional list of cell types to contrast against

    Raises:
        FileExistsError: If output directory already exists.
    """
    output_dir = Path(output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"Output directory {output_dir} already exists. Please delete it or use a different directory.")
    
    # TODO: download model if not exists
    # TODO: download metadata if not exists

    result = DecimaResult.load('tutorials/Supplementary_file_1.h5ad') # TODO: FIX implement loading from default metadata
    result.load_model('tutorials/rep0.ckpt')
    attributions = result.attributions(gene, cells, constract_cells)

    df_peaks = attributions.peaks
    df_peaks.to_csv(output_dir / "peaks.bed", sep="\t", header=False, index=False)

    attributions.plot_peaks()
    plt.savefig(output_dir / "peaks.png")

    with open(output_dir / "qc.log", "w") as f:
        f.write("TODO: write qc log based on the gene correlation")

    df_motifs = attributions.scan_motifs()
    df_motifs.to_csv(output_dir / "motifs.tsv", sep="\t", index=False)

    df_attributions = attributions.attrs
    df_attributions.to_csv(output_dir / "attributions_scores.tsv", sep="\t", index=False)

    for peak in df_peaks.itertuples():
        attributions.plot_attributions(relative_loc=peak.from_tss, window=50) # TODO: window to args
        plt.savefig(output_dir / f"attributions_{peak.peak}.png")

    attrs = attributions.attrs.sum(axis=0)
    attrs = attrs.reshape(1, -1)
    attrs = attrs.astype(np.float32)

    bw = pyBigWig.open(output_dir / "attributions.bigwig", "w")
    sizes = genomepy.Genome('hg38').sizes
    bw.addHeader([
        (chrom, size, 1)
        for chrom, size in sizes.items()
    ])
    bw.addEntries(attributions.chrom, [attributions.start], ends=[attributions.end], values=attrs)
    bw.close()

    