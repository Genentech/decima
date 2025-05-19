import numpy as np
import pandas as pd
import torch
import pyBigWig
import genomepy
from scipy.signal import find_peaks
from captum.attr import InputXGradient
from grelu.interpret.motifs import scan_sequences
from grelu.sequence.format import convert_input_type
from grelu.transforms.prediction_transforms import Aggregate, Specificity
from grelu.visualize import plot_attributions

from decima.plot.visualize import plot_attribution_peaks
from decima.data.read_hdf5 import extract_gene_data


def attributions(
    gene,
    tasks,
    model,
    device=None,
    h5_file=None,
    inputs=None,
    off_tasks=None,
    transform="specificity",
    method=InputXGradient,
    **kwargs,
):
    """Compute attributions for a gene.

    Args:
        gene: Gene symbol or ID to analyze
        tasks: List of cell types to analyze attributions for
        model: Model to use for attribution analysis
        device: Device to use for attribution analysis
        h5_file: Path to h5 file indexed by genes
        inputs: One-hot encoded sequence
        off_tasks: List of cell types to contrast against
        transform: Transformation to apply to attributions
        method: Method to use for attribution analysis

    Returns:
        Attribution: Attribution analysis results for the gene and tasks
    """

    if inputs is None:
        assert h5_file is not None
        inputs = extract_gene_data(h5_file, gene, merge=True)

    if transform == "specificity":
        model.add_transform(
            Specificity(
                on_tasks=tasks,
                off_tasks=off_tasks,
                model=model,
                compare_func="subtract",
            )
        )
    elif transform == "aggregate":
        model.add_transform(Aggregate(tasks=tasks, task_aggfunc="mean", model=model))

    model = model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    attributer = method(model.to(device))
    with torch.no_grad():
        attr = attributer.attribute(inputs.to(device), **kwargs).cpu().numpy()[:4]

    model.reset_transform()
    return attr


class Attribution:
    """
    Attribution analysis results for a gene.

    Args:
        gene: Gene symbol or ID to analyze
        inputs: One-hot encoded sequence
        attrs: Attribution scores
        chrom: Chromosome name
        start: Start position
        end: End position
        gene_start: Gene start position
        gene_end: Gene end position
        strand: Strand
        n_peaks: Number of peaks to find
        min_dist: Minimum distance between peaks

    Returns:
        Attribution: Attribution analysis results for the gene and tasks

    Examples:
        >>> attribution = Attribution(
            gene="A1BG",
            inputs=inputs,
            attrs=attrs,
            chrom="chr1",
            start=100,
            end=200,
            gene_start=100,
            gene_end=200,
            strand="+",
            n_peaks=10,
            min_dist=6
        )
        >>> attribution.plot_peaks()
        >>> attribution.scan_motifs()
        >>> attribution.plot_attributions()
        >>> attribution.save_bigwig(
        ...     "attributions.bigwig"
        ... )
        >>> attribution.peaks_to_bed()
    """

    def __init__(
        self,
        gene: str,
        inputs: torch.Tensor,
        attrs: np.ndarray,
        chrom: str,
        start: int,
        end: int,
        gene_start: int,
        gene_end: int,
        strand: str,
        n_peaks: int = 10,
        min_dist: int = 6,
    ):
        """Initialize Attribution.

        Args:
            inputs: One-hot encoded sequence
            preds: Model predictions
            attrs: Attribution scores
            chrom: Chromosome name
            start: Start position
            end: End position
            gene_start: Gene start position
            gene_end: Gene end position
            strand: Strand
            n_peaks: Number of peaks to find
            min_dist: Minimum distance between peaks
        """
        self.gene = gene
        self.inputs = inputs
        self.attrs = attrs
        self.chrom = chrom
        self.start = start
        self.end = end
        self.gene_start = gene_start
        self.gene_end = gene_end
        self.strand = strand
        self.relative_tss_pos = np.where(inputs[-1] == 1)[0][0]
        self.tss_pos = start + self.relative_tss_pos if strand == "+" else end - self.relative_tss_pos
        self.peaks = self._find_peaks(n_peaks, min_dist)

    def _find_peaks(self, n_peaks: int = 10, min_dist: int = 6):
        peaks, heights = find_peaks(self.attrs.sum(0), height=0.1, distance=min_dist)
        peaks = pd.DataFrame({"peak": peaks, "height": heights["peak_heights"]})
        peaks["from_tss"] = peaks["peak"] - self.relative_tss_pos
        return peaks.sort_values("height", ascending=False).head(n_peaks).reset_index(drop=True)

    def plot_peaks(self):
        """Plot attribution peaks."""
        return plot_attribution_peaks(self.attrs, self.relative_tss_pos)

    def scan_motifs(self, motifs: str = "hocomoco_v12", window: int = 18, pthresh: float = 5e-4) -> pd.DataFrame:
        """Scan for motifs in peak regions.

        Args:
            motifs: Motif database to use
            window: Window size around peaks
            pthresh: P-value threshold for motif matches

        Returns:
            pd.DataFrame: Motif scan results
        """
        peak_attrs = np.stack([self.attrs[:, peak - window // 2 : peak + window // 2] for peak in self.peaks.peak])
        peak_seqs = torch.stack([self.inputs[:4, peak - window // 2 : peak + window // 2] for peak in self.peaks.peak])

        results = scan_sequences(
            seqs=convert_input_type(peak_seqs, "strings"),
            motifs=motifs,
            pthresh=pthresh,
            rc=True,
            attrs=peak_attrs,
        )
        results["sequence"] = results["sequence"].astype(int)
        df = results.merge(self.peaks.reset_index(drop=True), left_on="sequence", right_index=True)
        df["seq_idx"] = self.gene + "@" + df["from_tss"].astype(str)
        return df.rename(columns={"seq_idx": "peak"})

    def plot_attributions(self, relative_loc=0, window=50, figsize=(10, 2)):
        """Plot attribution scores around a relative location.

        Args:
            relative_loc: Position relative to TSS to center plot on
            window: Number of bases to show on each side of center

        Returns:
            matplotlib.pyplot.Figure: Attribution plot
        """
        loc = self.relative_tss_pos + relative_loc
        return plot_attributions(self.attrs[:, loc - window : loc + window], figsize=figsize)

    def __repr__(self):
        return f"Attribution(gene={self.gene})"

    def __str__(self):
        return f"Attribution(gene={self.gene})"

    def save_bigwig(self, bigwig_path: str):
        """
        Save attribution scores as a bigwig file.

        Args:
            bigwig_path: Path to save bigwig file.
        """
        attrs = self.attrs.sum(axis=0)
        if self.strand == "-":
            attrs = attrs[::-1]

        bw = pyBigWig.open(bigwig_path, "w")

        sizes = genomepy.Genome("hg38").sizes
        bw.addHeader([(chrom, size) for chrom, size in sizes.items()])

        bw.addEntries(self.chrom, self.start, values=attrs, span=1, step=1)
        bw.close()

    def peaks_to_bed(self):
        """Convert peaks to bed format."""
        peaks = self.peaks.copy()
        peaks["chrom"] = self.chrom
        peaks["start"] = self.tss_pos + peaks["from_tss"] * (-1 if self.strand == "-" else 1) - 1
        peaks["end"] = peaks["start"] + 1

        peaks["name"] = self.gene + "@" + peaks["from_tss"].astype(str)
        peaks["strand"] = "."
        peaks["score"] = peaks["height"]
        return peaks[["chrom", "start", "end", "name", "score", "strand"]]

    def save_peaks(self, bed_path: str):
        """
        Save peaks to bed file.

        Args:
            bed_path: Path to save bed file.
        """
        self.peaks_to_bed().to_csv(bed_path, sep="\t", header=False, index=False)
