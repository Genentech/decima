from typing import List, Optional
import h5py
import numpy as np

from decima.core.result import DecimaResult


class AttributionResult:
    def __init__(
        self,
        attribution_h5: str,
        metadata_anndata: Optional[str] = None,
        tss_distance: Optional[int] = None,
        correct_grad=True,
    ):
        self.attribution_h5 = attribution_h5
        self.result = DecimaResult.load(metadata_anndata)
        self.tss_distance = tss_distance
        self.correct_grad = correct_grad

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

    def load(self, genes: List[str]):
        idx = [(self._idx[gene], gene) for gene in genes]
        sorted_idx, sorted_genes = zip(*sorted(idx, key=lambda x: x[0]))
        sorted_idx = list(sorted_idx)
        seqs = self.h5["sequence"][sorted_idx].astype(np.float32)
        attrs = self.h5["attribution"][sorted_idx].astype(np.float32)

        if self.correct_grad:
            # The following line applies a trick from Madjdandzic et al. to center the attributions.
            # By subtracting the mean attribution for each sequence, we ensure that the contributions of individual base
            # substitutions "speak for themselves." This prevents downstream tasks, like motif discovery, from being
            # influenced by the overall importance of a site rather than the specific mutational consequence of each base.
            attrs = attrs - attrs.mean(1, keepdims=True)

        seqs = {gene: seq for gene, seq in zip(sorted_genes, seqs)}
        attrs = {gene: attr for gene, attr in zip(sorted_genes, attrs)}

        if self.tss_distance is not None:
            tss_pos = self.result.gene_metadata.loc[genes, "gene_mask_start"].values
            window_start = tss_pos - self.tss_distance
            window_end = tss_pos + self.tss_distance

            seqs = np.array([seqs[gene][:, ws:we] for (gene, ws, we) in zip(genes, window_start, window_end)])
            attrs = np.array([attrs[gene][:, ws:we] for (gene, ws, we) in zip(genes, window_start, window_end)])
        else:
            seqs = np.array([seqs[gene] for gene in genes])
            attrs = np.array([attrs[gene] for gene in genes])

        return seqs, attrs

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
