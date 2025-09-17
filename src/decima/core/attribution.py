import warnings
from typing import List, Optional
import h5py
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from decima.constants import DECIMA_CONTEXT_SIZE
from decima.core.result import DecimaResult


class AttributionResult:
    def __init__(
        self,
        attribution_h5: str,
        metadata_anndata: Optional[str] = None,
        tss_distance: Optional[int] = None,
        correct_grad=True,
        num_workers: Optional[int] = -1,
    ):
        self.attribution_h5 = attribution_h5
        self.result = DecimaResult.load(metadata_anndata)
        self.tss_distance = tss_distance
        self.correct_grad = correct_grad
        self.num_workers = num_workers

    def open(self):
        self.h5 = h5py.File(self.attribution_h5, "r")
        self._idx = {gene.decode("utf-8"): i for i, gene in enumerate(self.h5["genes"][:])}
        self.genes = list(self._idx.keys())

        self.model_name = self.h5.attrs["model_name"]
        self.genome = self.h5.attrs["genome"]

    def close(self):
        self.h5.close()

    def __enter__(self):
        self.open()
        return self

    @staticmethod
    def _load(attribution_h5, idx: int, tss_pos: int, tss_distance: int, correct_grad: bool):
        with h5py.File(attribution_h5, "r") as f:
            # add padding to the left and right with length of tss distance
            padding = tss_distance or 0
            seqs = np.zeros((4, DECIMA_CONTEXT_SIZE + padding * 2))
            attrs = np.zeros((4, DECIMA_CONTEXT_SIZE + padding * 2))

            seqs[:, padding : DECIMA_CONTEXT_SIZE + padding] = f["sequence"][idx].astype(np.float32)
            attrs[:, padding : DECIMA_CONTEXT_SIZE + padding] = f["attribution"][idx].astype(np.float32)

        if tss_distance is not None:
            start = padding + tss_pos - tss_distance
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

    def load(self, genes: List[str]):
        tss_pos = self.result.gene_metadata.loc[genes, "gene_mask_start"].values

        if self.tss_distance is not None:
            padded_genes = [
                gene
                for gene, pos in zip(genes, tss_pos)
                if (pos + self.tss_distance > DECIMA_CONTEXT_SIZE) or (pos - self.tss_distance < 0)
            ]
            if len(padded_genes) > 0:
                warnings.warn(
                    f"Region of interest is greater than the context size and adding zero padding to genes:`{padded_genes}`."
                )

        seqs, attrs = zip(
            *Parallel(n_jobs=self.num_workers)(
                delayed(self._load)(self.attribution_h5, self._idx[gene], pos, self.tss_distance, self.correct_grad)
                for gene, pos in tqdm(
                    zip(genes, tss_pos), desc="Loading attributions and sequences...", total=len(genes)
                )
            )
        )

        return np.array(seqs), np.array(attrs)

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __repr__(self):
        return f"AttributionResult({self.attribution_h5})"

    def __str__(self):
        return f"AttributionResult({self.attribution_h5})"
