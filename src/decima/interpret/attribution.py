import numpy as np
import pandas as pd
import torch
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
    """Container for attribution analysis results."""
    
    def __init__(self, gene: str, inputs: torch.Tensor, attrs: np.ndarray, chrom: str, start: int, end: int, n_peaks: int = 10, min_dist: int = 6):
        """Initialize Attribution.
        
        Args:
            inputs: One-hot encoded sequence
            preds: Model predictions
            attrs: Attribution scores
            chrom: Chromosome name
            start: Start position
            end: End position
            n_peaks: Number of peaks to find
            min_dist: Minimum distance between peaks
        """
        self.gene = gene
        self.inputs = inputs
        self.attrs = attrs
        self.chrom = chrom
        self.start = start
        self.end = end
        self.tss_pos = np.where(inputs[-1] == 1)[0][0]
        self.peaks = self._find_peaks(n_peaks, min_dist)
    def _find_peaks(self, n_peaks: int = 10, min_dist: int = 6):
        peaks, heights = find_peaks(self.attrs.sum(0), height=0.1, distance=min_dist)
        peaks = pd.DataFrame({"peak": peaks, "height": heights["peak_heights"]})
        peaks["from_tss"] = peaks["peak"] - self.tss_pos
        peaks = peaks.sort_values("height", ascending=False).head(n_peaks)
        return peaks.reset_index(drop=True)

    def plot_peaks(self):
        """Plot attribution peaks."""
        return plot_attribution_peaks(self.attrs, self.tss_pos)

    def scan_motifs(self, motifs: str = 'hocomoco_v12', window: int = 18, pthresh: float = 5e-4) -> pd.DataFrame:
        """Scan for motifs in peak regions.
        
        Args:
            motifs: Motif database to use
            window: Window size around peaks
            pthresh: P-value threshold for motif matches
            
        Returns:
            pd.DataFrame: Motif scan results
        """
        # Get attributions and sequences for each peak region (window around peak)
        peak_attrs = np.stack([self.attrs[:, peak - window // 2 : peak + window // 2] for peak in self.peaks.peak])
        peak_seqs = torch.stack([self.inputs[:4, peak - window // 2 : peak + window // 2] for peak in self.peaks.peak])

        # Scan sequences for motifs in peak regions
        results = scan_sequences(
            seqs=convert_input_type(peak_seqs, "strings"),
            motifs=motifs,
            pthresh=pthresh,
            rc=True,
            attrs=peak_attrs,
        )
        results["sequence"] = results["sequence"].astype(int)
        return results.merge(self.peaks.reset_index(drop=True), left_on="sequence", right_index=True)

    def plot_attributions(self, relative_loc=0, window=50, figsize=(10, 2)):
        """Plot attribution scores around a relative location.
        
        Args:
            relative_loc: Position relative to TSS to center plot on
            window: Number of bases to show on each side of center
            
        Returns:
            matplotlib.pyplot.Figure: Attribution plot
        """
        loc = self.tss_pos + relative_loc
        return plot_attributions(self.attrs[:, loc - window:loc + window], figsize=figsize)

    def __repr__(self):
        return f"Attribution(gene={self.gene})"

    def __str__(self):
        return f"Attribution(gene={self.gene})"
