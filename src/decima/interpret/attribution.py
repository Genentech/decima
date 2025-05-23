from typing import Optional, Union
import warnings
import numpy as np
import pandas as pd
import torch
import pyBigWig
from pyfaidx import Faidx
import genomepy
from captum.attr import InputXGradient
from grelu.interpret.motifs import scan_sequences
from grelu.sequence.format import convert_input_type, strings_to_one_hot
from grelu.transforms.prediction_transforms import Aggregate, Specificity
from tangermeme.seqlet import recursive_seqlets

from decima.hub import load_decima_model
from decima.utils import get_compute_device
from decima.plot.visualize import plot_seqlogo, plot_peaks


def attributions(
    inputs,
    tasks,
    off_tasks=None,
    model=None,
    transform="specificity",
    method=InputXGradient,
    device=None,
    **kwargs,
):
    """Compute attributions for a gene.

    Args:
        gene: Gene symbol or ID to analyze
        tasks: List of cell types to analyze attributions for
        off_tasks: List of cell types to contrast against
        model: Model to use for attribution analysis
        device: Device to use for attribution analysis
        inputs: One-hot encoded sequence
        transform: Transformation to apply to attributions
        method: Method to use for attribution analysis

    Returns:
        Attribution: Attribution analysis results for the gene and tasks
    """
    if model is None:
        model = load_decima_model(model)

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
    device = get_compute_device(device)

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
        >>> attribution.plot_peaks()
        >>> attribution.save_bigwig(
        ...     "attributions.bigwig"
        ... )
        >>> attribution.peaks_to_bed()
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        attrs: np.ndarray,
        gene: Optional[str] = "",
        chrom: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        strand: Optional[str] = None,
        threshold: Optional[float] = 5e-4,
        min_seqlet_len: Optional[int] = 4,
        max_seqlet_len: Optional[int] = 25,
        additional_flanks: Optional[int] = 0,
    ):
        """Initialize Attribution.

        Args:
            inputs: One-hot encoded sequence
            attrs: Attribution scores
            gene: Gene name
            chrom: Chromosome name
            start: Start position
            end: End position
            gene_start: Gene start position
            gene_end: Gene end position
            strand: Strand
            threshold: Threshold for peak finding
            min_seqlet_len: Minimum sequence length for peak finding
            max_seqlet_len: Maximum sequence length for peak finding
            additional_flanks: Additional flanks to add to the gene
        """
        assert (
            inputs.shape[0] == 5
        ), "`inputs` must be 5-dimensional with shape (5, seq_len) where the last dimension is a binary mask."
        assert attrs.shape[0] == 4, "`attrs` must be 4-dimensional"
        assert inputs.shape[1] == attrs.shape[1], "`inputs` and `attrs` must have the same length"

        self.inputs = inputs
        self.attrs = attrs

        self.gene = gene
        self._chrom = chrom
        self._start = start
        self._end = end
        self._strand = strand
        assert self.end - self.start == self.inputs.shape[1], "`end` - `start` must be equal to the length of `inputs`"

        self.gene_mask_start = np.where(inputs[-1] == 1)[0][0]
        self.gene_mask_end = np.where(inputs[-1] == 1)[-1][0]

        self.peaks = self._find_peaks(
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
        )

    @property
    def chrom(self) -> str:
        """Get the chromosome name."""
        if self._chrom is None:
            return "custom"
        return self._chrom

    @property
    def start(self) -> int:
        """Get the start position."""
        if self._start is None:
            return 0
        return self._start

    @property
    def end(self) -> int:
        """Get the end position."""
        if self._end is None:
            return self.inputs.shape[1]
        return self._end

    @property
    def strand(self) -> str:
        """Get the strand."""
        if self._strand is None:
            return "+"
        return self._strand

    @property
    def gene_start(self) -> int:
        """Get the gene start position."""
        if self.strand == "-":
            return self.end - self.gene_mask_end
        return self.start + self.gene_mask_start

    @property
    def gene_end(self) -> int:
        """Get the gene end position."""
        if self.strand == "-":
            return self.end - self.gene_mask_start
        return self.start + self.gene_mask_end

    @classmethod
    def from_seq(
        cls,
        inputs,
        tasks,
        off_tasks,
        model: Optional[Union[str, int]] = 0,
        transform: str = "specificity",
        device: Optional[str] = None,
        gene: Optional[str] = "",
        chrom: Optional[str] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
        strand: Optional[str] = None,
        gene_mask_start: Optional[int] = None,
        gene_mask_end: Optional[int] = None,
        threshold: Optional[float] = 5e-4,
        min_seqlet_len: Optional[int] = 4,
        max_seqlet_len: Optional[int] = 25,
        additional_flanks: Optional[int] = 0,
    ):
        """Initialize Attribution from sequence.

        Args:
            inputs: Sequence to analyze
            tasks: List of cell types to analyze attributions for
            off_tasks: List of cell types to contrast against
            model: Model to use for attribution analysis
            transform: Transformation to apply to attributions
            device: Device to use for attribution analysis
            gene: Gene name
            chrom: Chromosome name
            start: Start position
            end: End position
            strand: Strand
            gene_start: Gene start position
            gene_end: Gene end position
            threshold: Threshold for peak finding
            min_seqlet_len: Minimum sequence length for peak finding
            max_seqlet_len: Maximum sequence length for peak finding
            additional_flanks: Additional flanks to add to the gene
        """
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).float()

        if isinstance(inputs, torch.Tensor):
            if (inputs.shape[0] == 4) and (gene_mask_start is not None) and (gene_mask_end is not None):
                mask = torch.zeros(1, 524288)
                mask[0, gene_mask_start:gene_mask_end] = 1.0
                inputs = torch.vstack([inputs, mask])
            elif inputs.shape[0] == 5:
                if (gene_mask_start is not None) or (gene_mask_end is not None):
                    warnings.warn("Gene mask will be ignored as sequence is 5-dimensional.")
                pass
            else:
                raise ValueError(
                    "Sequence must be 4-dimensional with shape (4, seq_len) "
                    "and gene start and end must be provided, or 5-dimensional "
                    "with shape (5, seq_len) where the last dimension is a binary mask."
                )
        elif isinstance(inputs, str):
            inputs = strings_to_one_hot(inputs, genome="hg38")
            assert (gene_mask_start is not None) and (
                gene_mask_end is not None
            ), "Gene start and end must be provided when seq is a string."
            mask = torch.zeros(1, 524288)
            mask[0, gene_mask_start:gene_mask_end] = 1.0
            inputs = torch.vstack([inputs, mask])
        else:
            raise ValueError("`inputs` must be a string, torch.Tensor, or np.ndarray")

        inputs = inputs.to(device)

        attrs = attributions(
            inputs=inputs,
            tasks=tasks,
            model=model,
            off_tasks=off_tasks,
            transform=transform,
            device=device,
        )
        return cls(
            inputs=inputs,
            attrs=attrs,
            gene=gene,
            chrom=chrom,
            start=start,
            end=end,
            strand=strand,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
        )

    @staticmethod
    def find_peaks(attrs, threshold=5e-4, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0):
        return recursive_seqlets(
            attrs.sum(0, keepdims=True),
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
        ).reset_index(drop=True)

    def _find_peaks(self, threshold=5e-4, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0):
        df = self.find_peaks(self.attrs, threshold, min_seqlet_len, max_seqlet_len, additional_flanks)
        del df["example_idx"]
        df["from_tss"] = df["start"] - self.gene_mask_start
        df["peak"] = self.gene + "@" + df["from_tss"].astype(str)
        return df[["peak", "start", "end", "attribution", "p-value", "from_tss"]]

    def scan_motifs(self, motifs: str = "hocomoco_v12", window: int = 18, pthresh: float = 5e-4) -> pd.DataFrame:
        """Scan for motifs in peak regions.

        Args:
            motifs: Motif database to use
            window: Window size around peaks
            pthresh: P-value threshold for motif matches

        Returns:
            pd.DataFrame: Motif scan results
        """
        mid = (self.peaks["start"] + self.peaks["end"]) // 2
        peak_attrs = np.stack([self.attrs[:, i - window : i + window] for i in mid])
        peak_seqs = torch.stack([self.inputs[:4, i - window : i + window] for i in mid])

        df = scan_sequences(
            seqs=convert_input_type(peak_seqs, "strings"),
            seq_ids=self.peaks["peak"].tolist(),
            motifs=motifs,
            pthresh=pthresh,
            rc=True,
            attrs=peak_attrs,
        ).rename(columns={"sequence": "peak"})

        df = df.merge(
            self.peaks[["peak", "from_tss", "start"]].reset_index(drop=True),
            on="peak",
            suffixes=("", "_peak"),
        )

        df["start"] += df["start_peak"] - window
        df["end"] += df["start_peak"] - window
        df["from_tss"] = df["start"] - self.gene_mask_start
        del df["start_peak"]
        del df["seq_idx"]

        return df.sort_values("p-value")

    def plot_peaks(self, overlapping_min_dist=1000, figsize=(10, 2)):
        """Plot attribution scores in a window around a relative location.

        Args:
            relative_loc: Position relative to TSS to center plot on
            window: Number of bases to show on each side of center
            figsize: Figure size in inches (width, height)

        Returns:
            matplotlib.pyplot.Figure: Attribution plot showing the window around the specified location
        """
        return plot_peaks(
            self.attrs,
            self.gene_mask_start,
            self.peaks,
            overlapping_min_dist=overlapping_min_dist,
            figsize=figsize,
        )

    def plot_seqlogo(self, relative_loc=0, window=50, figsize=(10, 2)):
        """Plot attribution scores around a relative location.

        Args:
            relative_loc: Position relative to TSS to center plot on
            window: Number of bases to show on each side of center

        Returns:
            matplotlib.pyplot.Figure: Attribution plot
        """
        loc = self.gene_mask_start + relative_loc
        return plot_seqlogo(self.attrs[:, loc - window : loc + window], figsize=figsize)

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

        if self._chrom is not None:
            sizes = genomepy.Genome("hg38").sizes
            bw.addHeader([(self.chrom, size) for chrom, size in sizes.items()])
        else:
            name = self.gene or "custom"
            bw.addHeader([(name, self.end - self.start)])

        bw.addEntries(self.chrom, self.start, values=attrs, span=1, step=1)
        bw.close()

    def save_fasta(self, fasta_path: str):
        """
        Save attribution scores as a fasta file.
        """
        seq = convert_input_type(self.inputs[:4], "strings")[0]
        name = self.gene or "custom"
        with open(fasta_path, "w") as f:
            f.write(f">{name}\n{seq}\n")
        Faidx(fasta_path, build_index=True)

    def peaks_to_bed(self):
        """
        Convert peaks to bed format.

        Returns:
            pd.DataFrame: Peaks in bed format where columns are:
                - chrom: Chromosome name
                - start: Start position in genome
                - end: End position in genome
                - name: Peak name in format "gene@from_tss"
                - score: Score (-log10(p-value)) clipped to 0-100 based on the seqlet calling
                - strand: Strand == '.'
        """
        df = self.peaks.copy().rename(columns={"peak": "name"})
        df["chrom"] = self.chrom

        if self.strand == "+":
            df["start"] = self.start + df["start"]
            df["end"] = self.start + df["end"]
        else:
            df["start"] = self.end - df["end"]
            df["end"] = self.end - df["start"]

        df["strand"] = "."
        df["score"] = -np.log10(df["p-value"] + 1e-100)
        df["score"] = df["score"].astype(int).clip(lower=0, upper=100)
        return df[["chrom", "start", "end", "name", "score", "strand"]]

    def save_peaks(self, bed_path: str):
        """
        Save peaks to bed file.

        Args:
            bed_path: Path to save bed file.
        """
        self.peaks_to_bed().to_csv(bed_path, sep="\t", header=False, index=False)
