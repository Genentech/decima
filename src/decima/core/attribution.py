"""
Attribution analysis from decima model.
"""

import warnings
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import h5py
import torch
import pyBigWig
from tqdm import tqdm
from pyfaidx import Faidx
from joblib import Parallel, delayed
from tangermeme.seqlet import recursive_seqlets

from grelu.interpret.motifs import scan_sequences
from grelu.sequence.format import convert_input_type, strings_to_one_hot
from grelu.visualize import plot_attributions

from decima.constants import DECIMA_CONTEXT_SIZE, DEFAULT_ENSEMBLE, MODEL_METADATA
from decima.core.result import DecimaResult
from decima.interpret.attributer import DecimaAttributer
from decima.utils.sequence import one_hot_to_seq
from decima.plot.visualize import plot_peaks


class Attribution:
    """
    Attribution analysis results for a gene.

    Args:
        gene: Gene symbol or ID to analyze
        inputs: One-hot encoded sequence
        attrs: Attribution scores
        gene: Gene name
        chrom: Chromosome name
        start: Start position
        end: End position
        strand: Strand
        threshold: Threshold for peak finding
        min_seqlet_len: Minimum sequence length for peak finding
        max_seqlet_len: Maximum sequence length for peak finding
        additional_flanks: Additional flanks to add to the gene

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
            strand="+",
            threshold=5e-4,
            min_seqlet_len=4,
            max_seqlet_len=25,
            additional_flanks=0,
        )
        >>> attribution.plot_peaks()
        >>> attribution.scan_motifs()
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
        pattern_type: Optional[str] = "both",
    ):
        """Initialize Attribution.

        Args:
            inputs: One-hot encoded sequence
            attrs: Attribution scores
            gene: Gene name
            chrom: Chromosome name
            start: Start position
            end: End position
            strand: Strand
            threshold: Threshold for peak finding
            min_seqlet_len: Minimum sequence length for peak finding
            max_seqlet_len: Maximum sequence length for peak finding
            additional_flanks: Additional flanks to add to the gene
            pattern_type: Pattern type to use for peak finding default is "both", alternatively "pos" or "neg" which will only consider positive or negative peaks respectively.
              "both" means both positive and negative patterns are considered.
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
        self._peaks = None
        assert self.end - self.start == self.inputs.shape[1], "`end` - `start` must be equal to the length of `inputs`"

        self.gene_mask_start = np.where(inputs[-1] == 1)[0][0]
        self.gene_mask_end = np.where(inputs[-1] == 1)[-1][0]

        # Recursive seqlet calling parameters
        self.threshold = threshold
        self.min_seqlet_len = min_seqlet_len
        self.max_seqlet_len = max_seqlet_len
        self.additional_flanks = additional_flanks
        self.pattern_type = pattern_type

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

    @property
    def peaks(self) -> pd.DataFrame:
        if self._peaks is None:
            self._peaks = self._find_peaks(
                threshold=self.threshold,
                min_seqlet_len=self.min_seqlet_len,
                max_seqlet_len=self.max_seqlet_len,
                additional_flanks=self.additional_flanks,
            )
        return self._peaks

    @classmethod
    def from_seq(
        cls,
        inputs: Union[str, torch.Tensor, np.ndarray],
        tasks: Optional[list] = None,
        off_tasks: Optional[list] = None,
        model: Optional[Union[str, int]] = MODEL_METADATA[DEFAULT_ENSEMBLE][0],
        transform: str = "specificity",
        method: str = "inputxgradient",
        device: Optional[str] = "cpu",
        result: Optional[str] = None,
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
            inputs: Sequence to analyze either string of sequence,
                torch.Tensor or np.ndarray with shape (4, 524288)
                or (5, 524288) where the last dimension is a binary mask.
                If 4-dimensional, gene_mask_start and gene_mask_end must be provided.
            tasks: List of cell types to analyze attributions for
            off_tasks: List of cell types to contrast against
            model: Model to use for attribution analysis
            transform: Transformation to apply to attributions
            method: Method to use for attribution analysis available options: "saliency", "inputxgradient", "integratedgradients".
            device: Device to use for attribution analysis
            result: Result object or path to result object or name of the model to load the result for.
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
                mask = torch.zeros(1, DECIMA_CONTEXT_SIZE)
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
            inputs = strings_to_one_hot(inputs)
            assert (gene_mask_start is not None) and (
                gene_mask_end is not None
            ), "Gene start and end must be provided when seq is a string."
            mask = torch.zeros(1, DECIMA_CONTEXT_SIZE)
            mask[0, gene_mask_start:gene_mask_end] = 1.0
            inputs = torch.vstack([inputs, mask])
        else:
            raise ValueError("`inputs` must be a string, torch.Tensor, or np.ndarray")

        result = DecimaResult.load(result or model)
        tasks, off_tasks = result.query_tasks(tasks, off_tasks)

        attrs = (
            DecimaAttributer.load_decima_attributer(
                model_name=model,
                tasks=tasks,
                off_tasks=off_tasks,
                transform=transform,
                method=method,
                device=device,
            )
            .attribute(inputs=inputs.unsqueeze(0))
            .squeeze(0)
            .cpu()
            .numpy()
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
    def find_peaks(
        attrs, threshold=5e-4, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, pattern_type="both"
    ):
        """Find peaks in attribution scores.

        Args:
            attrs: Attribution scores
            threshold: Threshold for peak finding
            min_seqlet_len: Minimum sequence length for peak finding
            max_seqlet_len: Maximum sequence length for peak finding
            additional_flanks: Additional flanks to add to the gene
            pattern_type: Pattern type to use for peak finding default is "both", alternatively "pos" or "neg" which will only consider positive or negative peaks respectively.

        Returns:
            df: DataFrame of peaks with columns of:
                - "peak": Peak name in format "pattern_type.gene@from_tss"
                - "start": Start position of the peak
                - "end": End position of the peak
                - "attribution": Attribution score of the peak
                - "p-value": P-value of the peak
                - "from_tss": Distance from the TSS to the peak
                - "pattern_type": Pattern type of the peak
        """
        attrs = attrs.sum(0, keepdims=True)
        if pattern_type == "both":
            return pd.concat(
                [
                    Attribution.find_peaks(attrs, threshold, min_seqlet_len, max_seqlet_len, additional_flanks, "pos"),
                    Attribution.find_peaks(attrs, threshold, min_seqlet_len, max_seqlet_len, additional_flanks, "neg"),
                ]
            )
        elif pattern_type == "pos":
            pass
        elif pattern_type == "neg":
            attrs = -attrs
        else:
            raise ValueError(f"Invalid pattern type: {pattern_type}")

        df = (
            recursive_seqlets(
                attrs.sum(0, keepdims=True),
                threshold=threshold,
                min_seqlet_len=min_seqlet_len,
                max_seqlet_len=max_seqlet_len,
                additional_flanks=additional_flanks,
            )
            .reset_index(drop=True)
            .assign(pattern_type=pattern_type)
            .query("attribution > 0")
        )
        if pattern_type == "neg":
            df["attribution"] = -df["attribution"]
        return df

    def _find_peaks(
        self, threshold=5e-4, min_seqlet_len=4, max_seqlet_len=25, additional_flanks=0, pattern_type="both"
    ):
        df = self.find_peaks(self.attrs, threshold, min_seqlet_len, max_seqlet_len, additional_flanks, pattern_type)
        del df["example_idx"]
        df["from_tss"] = df["start"] - self.gene_mask_start
        df["peak"] = df["pattern_type"] + "." + self.gene + "@" + df["from_tss"].astype(str)
        return df[["peak", "start", "end", "attribution", "p-value", "from_tss"]]

    def _zero_pad(self, array: np.ndarray, start: int, end: int):
        prefix_pad = max(0, -start)
        suffix_pad = max(0, end - array.shape[1])
        start = max(0, start)
        end = min(end, array.shape[1])
        return np.pad(array[:, start:end], ((0, 0), (prefix_pad, suffix_pad)), "constant", constant_values=0)

    def _get_inputs(self, start: int, end: int):
        return self._zero_pad(self.inputs, start, end)

    def _get_attrs(self, start: int, end: int):
        return self._zero_pad(self.attrs, start, end)

    def scan_motifs(self, motifs: str = "hocomoco_v13", window: int = 18, pthresh: float = 5e-4) -> pd.DataFrame:
        """Scan for motifs in peak regions.

        Args:
            motifs: Motif database to use
            window: Window size around peaks
            pthresh: P-value threshold for motif matches

        Returns:
            pd.DataFrame: Motif scan results with columns:
                - "motif": Motif name
                - "peak": Peak name
                - "start": Start position of the peak
                - "end": End position of the peak
                - "strand": Strand of the peak
                - "score": Fimoe score of the motif
                - "p-value": Fimo p-value of the motif
                - "matched_seq": Matched sequence
                - "site_attr_score": Site attribution score
                - "motif_attr_score": Motif attribution score
                - "from_tss": Distance from the TSS to the peak
        """
        mid = (self.peaks["start"] + self.peaks["end"]) // 2

        peak_attrs = np.stack([self._get_attrs(i - window, i + window) for i in mid])
        peak_seqs = [one_hot_to_seq(self._get_inputs(i - window, i + window)) for i in mid]

        df = scan_sequences(
            seqs=peak_seqs,
            seq_ids=self.peaks["peak"].tolist(),
            motifs=motifs,
            pthresh=pthresh,
            rc=True,
            attrs=peak_attrs,
        ).rename(columns={"sequence": "peak", "fimo_p-value": "p-value", "fimo_score": "score"})

        df = df.merge(
            self.peaks[["peak", "from_tss", "start"]].assign(mid=mid).reset_index(drop=True),
            on="peak",
            suffixes=("", "_peak"),
        )

        df["start"] += df["mid"] - window
        df["end"] += df["mid"] - window
        df["from_tss"] = df["start"] - self.gene_mask_start
        del df["start_peak"]
        del df["seq_idx"]
        del df["mid"]

        return df.sort_values("p-value")

    def plot_peaks(self, overlapping_min_dist=1000, figsize=(10, 2)):
        """Plot attribution scores and highlight peaks.

        Args:
            overlapping_min_dist: Minimum distance between peaks to consider them overlapping
            figsize: Figure size in inches (width, height)

        Returns:
            plotnine.ggplot: The plotted figure showing attribution scores with highlighted peaks
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

        if self._chrom is not None:
            name = self.chrom
            import genomepy

            sizes = genomepy.Genome("hg38").sizes
            bw.addHeader([(chrom, size) for chrom, size in sizes.items()])
        else:
            name = self.gene or "custom"
            bw.addHeader([(name, self.end - self.start)])

        bw.addEntries(name, self.start, values=attrs, span=1, step=1)
        bw.close()

    def fasta_str(self):
        """
        Get attribution scores as a fasta string.
        """
        seq = convert_input_type(self.inputs[:4], "strings")
        name = self.gene or "custom"
        return f">{name}\n{seq}\n"

    def save_fasta(self, fasta_path: str):
        """
        Save attribution scores as a fasta file.
        """
        with open(fasta_path, "w") as f:
            f.write(self.fasta_str())
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
            df["start"], df["end"] = self.start + df["start"], self.start + df["end"]
        else:
            df["start"], df["end"] = self.end - df["end"], self.end - df["start"]

        df["strand"] = "."
        # np.maximum because of https://github.com/jmschrei/tangermeme/issues/40
        df["score"] = -np.log10(np.maximum(df["p-value"], 0) + 1e-50)
        df["score"] = df["score"].round(5).clip(lower=0, upper=50)
        return df[["chrom", "start", "end", "name", "score", "strand", "attribution"]].sort_values(["chrom", "start"])

    def save_peaks(self, bed_path: str):
        """
        Save peaks to bed file.

        Args:
            bed_path: Path to save bed file.
        """
        self.peaks_to_bed().to_csv(bed_path, sep="\t", header=False, index=False)

    def __sub__(self, other):
        assert self.chrom == other.chrom, "Chromosomes must be the same to subtract attributions."
        assert self.start == other.start, "Starts must be the same to subtract attributions."
        assert self.end == other.end, "Ends must be the same to subtract attributions."
        assert self.strand == other.strand, "Strands must be the same to subtract attributions."

        if (
            (self.threshold != other.threshold)
            or (self.min_seqlet_len != other.min_seqlet_len)
            or (self.max_seqlet_len != other.max_seqlet_len)
            or (self.additional_flanks != other.additional_flanks)
            or (self.pattern_type != other.pattern_type)
        ):
            warnings.warn(
                "`threshold`, `min_seqlet_len`, `max_seqlet_len`, `additional_flanks`, and `pattern_type` are not the same, overriding "
                "them with the values of the first attribution object."
            )

        return Attribution(
            inputs=self.inputs,
            attrs=self.attrs - other.attrs,
            gene=f"{self.gene}-{other.gene}",
            chrom=self.chrom,
            start=self.start,
            end=self.end,
            strand=self.strand,
            threshold=self.threshold,
            min_seqlet_len=self.min_seqlet_len,
            max_seqlet_len=self.max_seqlet_len,
            additional_flanks=self.additional_flanks,
            pattern_type=self.pattern_type,
        )


class AttributionResult:
    """
    Attribution result from decima model.

    Args:
        attribution_h5: Path to attribution h5 file or list of paths to attribution h5 files generated by `decima attributions-predict` or `decima attributions` commands.
        tss_distance: Distance from the TSS to include in the attribution analysis.
        correct_grad: Whether to correct the gradient for the attribution analysis.
        num_workers: Number of workers to use for the attribution analysis.
        agg_func: Function to aggregate the attribution scores.


    Examples:
        with AttributionResult(attribution_h5=["example/attribution.h5", "example/attribution2.h5"]) as ar:
            seqs, attrs = ar.load(genes=["SPI1"])
            attribution = ar.load_attribution(gene="SPI1")
    """

    def __init__(
        self,
        attribution_h5: Union[str, List[str]],
        tss_distance: Optional[int] = None,
        correct_grad=True,
        num_workers: Optional[int] = -1,
        agg_func: Optional[str] = None,
    ):
        self.attribution_h5 = attribution_h5
        self.tss_distance = tss_distance
        self.correct_grad = correct_grad
        self.num_workers = num_workers
        self.agg_func = agg_func

    def open(self):
        """Open the attribution h5 files."""
        if isinstance(self.attribution_h5, list):
            self.h5 = [h5py.File(str(attribution_h5), "r") for attribution_h5 in self.attribution_h5]

            for i, attribution_h5_file in enumerate(self.attribution_h5):
                with h5py.File(str(attribution_h5_file), "r") as f:
                    if i == 0:
                        self.genes = f["genes"][:].astype("U100")
                        self.genome = f.attrs["genome"]
                    else:
                        assert all(self.genes == f["genes"][:].astype("U100")), (
                            "All genes must be the same in all attribution files. "
                            f"Expected: {self.genes}, Found: {f['genes'][:].astype('U100')}"
                        )
                        assert (
                            self.genome == f.attrs["genome"]
                        ), "All attribution files must have the same genome version."
            self.model_name = list()
            for h5 in self.h5:
                self.model_name.append(h5.attrs["model_name"])
        else:
            self.h5 = h5py.File(str(self.attribution_h5), "r")
            self.model_name = self.h5.attrs["model_name"]
            self.genome = self.h5.attrs["genome"]
            self.genes = self.h5["genes"][:].astype("U100")

        self._idx = {gene: i for i, gene in enumerate(self.genes)}

    def close(self):
        """Close the attribution h5 files."""
        if isinstance(self.attribution_h5, list):
            for h5 in self.h5:
                h5.close()
        else:
            self.h5.close()

    def __enter__(self):
        self.open()
        return self

    @staticmethod
    def aggregate(seqs, attrs, agg_func: Optional[str] = None):
        """Aggregate the attribution scores."""
        if agg_func is None:
            return seqs, attrs
        elif agg_func == "mean":
            return np.mean(seqs, axis=0), np.mean(attrs, axis=0)
        elif agg_func == "sum":
            return np.sum(seqs, axis=0), np.sum(attrs, axis=0)
        else:
            raise ValueError(f"Invalid aggregation function: {agg_func}")

    @staticmethod
    def _load(
        attribution_h5,
        idx: int,
        tss_distance: int,
        correct_grad: bool,
        gene_mask: bool = False,
        sequence_key: str = "sequence",
        attribution_key: str = "attribution",
    ):
        """Load the attribution scores."""
        with h5py.File(str(attribution_h5), "r") as f:
            gene = f["genes"][idx].decode("utf-8")
            gene_mask_start = f["gene_mask_start"][idx].astype(int)
            gene_mask_end = f["gene_mask_end"][idx].astype(int)

            if tss_distance is not None:
                if (gene_mask_start + tss_distance > DECIMA_CONTEXT_SIZE) or (gene_mask_start - tss_distance < 0):
                    warnings.warn(
                        f"Window around the TSS is greater than the context size and adding zero padding to `{gene}`"
                        f" where window around the TSS: `{gene_mask_start} ± {tss_distance}`."
                        f" The context size of decima is `{DECIMA_CONTEXT_SIZE}`."
                    )

            padding = tss_distance or 0
            if gene_mask:
                seqs = np.zeros((5, DECIMA_CONTEXT_SIZE + padding * 2))
                seqs[-1, padding + gene_mask_start : padding + gene_mask_end] = 1
            else:
                seqs = np.zeros((4, DECIMA_CONTEXT_SIZE + padding * 2))

            seqs[:4, padding : DECIMA_CONTEXT_SIZE + padding] = convert_input_type(
                f[sequence_key][idx].astype("int"), "one_hot", input_type="indices"
            )

            attrs = np.zeros((4, DECIMA_CONTEXT_SIZE + padding * 2))
            attrs[:, padding : DECIMA_CONTEXT_SIZE + padding] = f[attribution_key][idx].astype(np.float32)

        if tss_distance is not None:
            start = padding + gene_mask_start - tss_distance
            end = start + tss_distance * 2

            seqs = seqs[:, start:end]
            attrs = attrs[:, start:end]

        if correct_grad:
            # The following line applies a trick from Madjdandzic et al. to center the attributions.
            # By subtracting the mean attribution for each sequence, we ensure that the contributions of individual base
            # substitutions "speak for themselves." This prevents downstream tasks, like motif discovery, from being
            # influenced by the overall importance of a site rather than the specific mutational consequence of each base.
            attrs = attrs - attrs.mean(0, keepdims=True)

        return seqs, attrs

    @staticmethod
    def _load_multiple(
        attribution_h5_files,
        idx: int,
        tss_distance: int,
        correct_grad: bool,
        gene_mask: bool = False,
        agg_func: Optional[str] = None,
        sequence_key: str = "sequence",
        attribution_key: str = "attribution",
    ):
        """Load the attribution scores from multiple attribution h5 files."""
        seqs, attrs = zip(
            *(
                AttributionResult._load(
                    attribution_h5_file, idx, tss_distance, correct_grad, gene_mask, sequence_key, attribution_key
                )
                for attribution_h5_file in attribution_h5_files
            )
        )
        return AttributionResult.aggregate(np.array(seqs), np.array(attrs), agg_func)

    def load(self, genes: List[str], gene_mask: bool = False, **kwargs):
        """Load the attribution scores for a list of genes.

        Args:
            genes: List of genes to load.
            gene_mask: Whether to mask the gene.

        Returns:
            seqs: Array of sequences.
            attrs: Array of attribution scores.
        """
        if getattr(self, "h5", None) is None:
            raise ValueError(
                "AttributionResult is not open. Please open it with `with AttributionResult(attribution_h5) as ar:`."
            )

        load_func = self._load
        load_kwargs = {
            "tss_distance": self.tss_distance,
            "correct_grad": self.correct_grad,
            "gene_mask": gene_mask,
            **kwargs,
        }
        if isinstance(self.attribution_h5, list):
            load_func = self._load_multiple
            load_kwargs["agg_func"] = self.agg_func

        seqs, attrs = zip(
            *Parallel(n_jobs=self.num_workers)(
                delayed(load_func)(self.attribution_h5, self._idx[gene], **load_kwargs)
                for gene in tqdm(genes, desc="Loading attributions and sequences...")
            )
        )
        return np.array(seqs), np.array(attrs)

    @staticmethod
    def _load_attribution(
        attribution_h5: Union[str, List[str]],
        idx: int,
        gene: str,
        tss_distance: int,
        chrom: str,
        start: int,
        end: int,
        agg_func: Optional[str] = None,
        threshold: float = 5e-4,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        additional_flanks: int = 0,
        pattern_type: str = "both",
        sequence_key: str = "sequence",
        attribution_key: str = "attribution",
        differential: bool = False,
        alt_sequence_key: str = "sequence_alt",
        alt_attribution_key: str = "attribution_alt",
    ):
        kwargs = {
            "tss_distance": tss_distance,
            "correct_grad": False,
            "gene_mask": True,
            "sequence_key": sequence_key,
            "attribution_key": attribution_key,
        }
        if isinstance(attribution_h5, list):
            assert agg_func is not None, "Aggregation function must be set to use recursive seqlet calling."
            seqs, attrs = AttributionResult._load_multiple(attribution_h5, idx, agg_func=agg_func, **kwargs)
        else:
            seqs, attrs = AttributionResult._load(attribution_h5, idx, **kwargs)

        attribution = Attribution(
            inputs=seqs,
            attrs=attrs,
            gene=gene,
            chrom=chrom,
            start=start,
            end=end,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
            pattern_type=pattern_type,
        )

        if not differential:
            return attribution
        else:
            attribution_alt = AttributionResult._load_attribution(
                attribution_h5,
                idx,
                gene,
                tss_distance,
                chrom,
                start,
                end,
                agg_func,
                threshold=threshold,
                min_seqlet_len=min_seqlet_len,
                max_seqlet_len=max_seqlet_len,
                additional_flanks=additional_flanks,
                pattern_type=pattern_type,
                attribution_key=alt_attribution_key,
                sequence_key=alt_sequence_key,
                differential=False,
            )
            return attribution_alt - attribution

    def _get_metadata(self, idx: List[str], metadata_anndata: Optional[str] = None, custom_genome: bool = False):
        genes = self.genes[idx]
        if custom_genome:
            chroms = genes
            starts = [0] * len(genes)
            if self.tss_distance is not None:
                ends = [self.tss_distance * 2] * len(genes)
            else:
                ends = [DECIMA_CONTEXT_SIZE] * len(genes)
        else:
            model_name = self.model_name
            if isinstance(model_name, list):
                model_name = model_name[0]
            result = DecimaResult.load(metadata_anndata or model_name)
            chroms = result.gene_metadata.loc[genes].chrom
            if self.tss_distance is not None:
                tss_pos = np.where(
                    result.gene_metadata.loc[genes].strand == "-",
                    result.gene_metadata.loc[genes].gene_end,
                    result.gene_metadata.loc[genes].gene_start,
                )
                starts = tss_pos - self.tss_distance
                ends = tss_pos + self.tss_distance
            else:
                starts = result.gene_metadata.loc[genes].start
                ends = result.gene_metadata.loc[genes].end
        return chroms, starts, ends

    def load_attribution(
        self,
        gene: str,
        metadata_anndata: Optional[str] = None,
        custom_genome: bool = False,
        threshold: float = 5e-4,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        additional_flanks: int = 0,
        pattern_type: str = "both",
        **kwargs,
    ):
        """Load the attribution scores for a gene.

        Args:
            gene: Gene to load.
            metadata_anndata: Metadata anndata object.
            custom_genome: Whether to use custom genome.
            threshold: Threshold for peak finding.
            min_seqlet_len: Minimum sequence length for peak finding.
            max_seqlet_len: Maximum sequence length for peak finding.
            additional_flanks: Additional flanks to add to the gene.
            pattern_type: Pattern type to use for peak finding default is "both", alternatively "pos" or "neg" which will only consider positive or negative peaks respectively.

        Returns:
            Attribution object.
        """
        idx = self._idx[gene]
        chroms, starts, ends = self._get_metadata(idx, metadata_anndata, custom_genome)
        return self._load_attribution(
            self.attribution_h5,
            idx,
            self.genes[idx],
            self.tss_distance,
            chroms,
            starts,
            ends,
            self.agg_func,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
            pattern_type=pattern_type,
            **kwargs,
        )

    @staticmethod
    def _recursive_seqlet_calling(
        attribution_h5: Union[str, List[str]],
        idx: int,
        gene: str,
        tss_distance: int,
        chrom: str,
        start: int,
        end: int,
        agg_func: Optional[str] = None,
        threshold: float = 5e-4,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        additional_flanks: int = 0,
        pattern_type: str = "both",
        meme_motif_db: str = "hocomoco_v13",
        **kwargs,
    ):
        attribution = AttributionResult._load_attribution(
            attribution_h5,
            idx,
            gene,
            tss_distance,
            chrom,
            start,
            end,
            agg_func,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
            pattern_type=pattern_type,
            **kwargs,
        )
        df_peaks = attribution.peaks_to_bed()
        df_motifs = attribution.scan_motifs(motifs=meme_motif_db)
        return df_peaks, df_motifs

    def recursive_seqlet_calling(
        self,
        genes: Optional[List[str]] = None,
        metadata_anndata: Optional[str] = None,
        custom_genome: bool = False,
        threshold: float = 5e-4,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        additional_flanks: int = 0,
        pattern_type: str = "both",
        meme_motif_db: str = "hocomoco_v13",
        **kwargs,
    ):
        """Perform recursive seqlet calling on the attribution scores.

        Args:
            genes: List of genes to perform recursive seqlet calling on.
            metadata_anndata: Metadata anndata object.
            custom_genome: Whether to use custom genome.
            threshold: Threshold for peak finding.
            min_seqlet_len: Minimum sequence length for peak finding.
            max_seqlet_len: Maximum sequence length for peak finding.
            additional_flanks: Additional flanks to add to the gene.
            pattern_type: Pattern type to use for peak finding default is "both", alternatively "pos" or "neg" which will only consider positive or negative peaks respectively.
            meme_motif_db: MEME motif database to use for motif discovery.

        Returns:
            df_peaks: DataFrame of peaks.
            df_motifs: DataFrame of motifs.
        """
        if genes is None:
            genes = self.genes

        chroms, starts, ends = self._get_metadata([self._idx[gene] for gene in genes], metadata_anndata, custom_genome)

        df_peaks, df_motifs = zip(
            *Parallel(n_jobs=self.num_workers)(
                delayed(AttributionResult._recursive_seqlet_calling)(
                    self.attribution_h5,
                    self._idx[gene],
                    gene if isinstance(gene, str) else "_".join(gene),
                    self.tss_distance,
                    chrom,
                    start,
                    end,
                    self.agg_func,
                    threshold=threshold,
                    min_seqlet_len=min_seqlet_len,
                    max_seqlet_len=max_seqlet_len,
                    additional_flanks=additional_flanks,
                    pattern_type=pattern_type,
                    meme_motif_db=meme_motif_db,
                    **kwargs,
                )
                for gene, chrom, start, end in tqdm(
                    zip(genes, chroms, starts, ends), desc="Computing recursive seqlet calling...", total=len(genes)
                )
            )
        )
        return pd.concat(df_peaks).reset_index(drop=True), pd.concat(df_motifs).reset_index(drop=True)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"AttributionResult({self.attribution_h5})"


class VariantAttributionResult(AttributionResult):
    def __init__(
        self,
        attribution_h5: Union[str, List[str]],
        tss_distance: Optional[int] = None,
        correct_grad: bool = True,
        num_workers: Optional[int] = -1,
        agg_func: Optional[str] = None,
    ):
        super().__init__(attribution_h5, tss_distance, correct_grad, num_workers, agg_func)

    def open(self):
        super().open()
        if isinstance(self.attribution_h5, list):
            for i, attribution_h5_file in enumerate(self.attribution_h5):
                with h5py.File(str(attribution_h5_file), "r") as f:
                    if i == 0:
                        self.genes = f["genes"][:].astype("U100")
                        self.variants = f["variants"][:].astype("U100")
                        self.rel_pos = f["rel_pos"][:].astype(int)
                    else:
                        assert all(self.genes == f["genes"][:].astype("U100")), (
                            "All genes must be the same in all attribution files. "
                            f"Expected: {self.genes}, Found: {f['genes'][:].astype('U100')}"
                        )
                        assert all(self.variants == f["variants"][:].astype("U100")), (
                            "All variants must be the same in all attribution files. "
                            f"Expected: {self.variants}, Found: {f['variants'][:].astype('U100')}"
                        )
            self._idx = {(variant, gene): i for i, (variant, gene) in enumerate(zip(self.variants, self.genes))}
            gene_mask_start = self.h5[0]["gene_mask_start"][:].astype(int)
        else:
            self.genes = self.h5["genes"][:].astype("U100")
            self.variants = self.h5["variants"][:].astype("U100")
            self.rel_pos = self.h5["rel_pos"][:].astype(int)
            gene_mask_start = self.h5["gene_mask_start"][:].astype(int)

        self.df_variants = pd.DataFrame(
            {
                "variant": self.variants,
                "gene": self.genes,
                "rel_pos": self.rel_pos,
                "tss_dist": self.rel_pos - gene_mask_start,
            }
        )
        self._idx = {(variant, gene): i for i, (variant, gene) in enumerate(zip(self.variants, self.genes))}

    def load(self, variants: List[str], genes: List[str], gene_mask: bool = False):
        """Load the attribution scores for a list of genes and variants."""
        variant_gene = list(zip(variants, genes))
        seqs_ref, attrs_ref = super().load(
            variant_gene, gene_mask, sequence_key="sequence", attribution_key="attribution"
        )
        seqs_alt, attrs_alt = super().load(
            variant_gene, gene_mask, sequence_key="sequence_alt", attribution_key="attribution_alt"
        )
        return seqs_ref, attrs_ref, seqs_alt, attrs_alt

    def load_attribution(
        self,
        variant: str,
        gene: str,
        metadata_anndata: Optional[str] = None,
        custom_genome: bool = False,
        threshold: float = 5e-4,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        additional_flanks: int = 0,
        pattern_type: str = "both",
        **kwargs,
    ):
        idx = self._idx[(variant, gene)]
        chroms, starts, ends = self._get_metadata(idx, metadata_anndata, custom_genome)
        attribution_ref = self._load_attribution(
            self.attribution_h5,
            idx,
            gene,
            self.tss_distance,
            chroms,
            starts,
            ends,
            self.agg_func,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
            pattern_type=pattern_type,
            sequence_key="sequence",
            attribution_key="attribution",
        )
        attribution_alt = self._load_attribution(
            self.attribution_h5,
            idx,
            f"{variant}_{gene}",
            self.tss_distance,
            chroms,
            starts,
            ends,
            self.agg_func,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
            pattern_type=pattern_type,
            sequence_key="sequence_alt",
            attribution_key="attribution_alt",
        )
        return attribution_ref, attribution_alt

    def recursive_seqlet_calling(
        self,
        variants: List[str],
        genes: Optional[List[str]],
        metadata_anndata: Optional[str] = None,
        threshold: float = 5e-4,
        min_seqlet_len: int = 4,
        max_seqlet_len: int = 25,
        additional_flanks: int = 0,
        pattern_type: str = "both",
        meme_motif_db: str = "hocomoco_v13",
    ):
        variant_gene = list(zip(variants, genes))

        df_peaks_ref, df_motifs_ref = super().recursive_seqlet_calling(
            variant_gene,
            metadata_anndata,
            custom_genome=False,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
            pattern_type=pattern_type,
            meme_motif_db=meme_motif_db,
            sequence_key="sequence",
            attribution_key="attribution",
        )
        df_peaks_alt, df_motifs_alt = super().recursive_seqlet_calling(
            variant_gene,
            metadata_anndata,
            custom_genome=False,
            threshold=threshold,
            min_seqlet_len=min_seqlet_len,
            max_seqlet_len=max_seqlet_len,
            additional_flanks=additional_flanks,
            pattern_type=pattern_type,
            meme_motif_db=meme_motif_db,
            sequence_key="sequence_alt",
            attribution_key="attribution_alt",
        )
        df_peaks = pd.concat([df_peaks_ref.assign(allele="ref"), df_peaks_alt.assign(allele="alt")]).reset_index(
            drop=True
        )
        df_motifs = pd.concat([df_motifs_ref.assign(allele="ref"), df_motifs_alt.assign(allele="alt")]).reset_index(
            drop=True
        )
        return df_peaks, df_motifs
