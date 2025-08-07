from typing import List
import h5py
import numpy as np

from decima.core.result import DecimaResult


class AttributionResult:
    def __init__(self, attribution_h5: str, metadata_anndata: str):
        self.attribution_h5 = attribution_h5
        self.result = DecimaResult.load(metadata_anndata)

    def open(self):
        self.h5 = h5py.File(self.attribution_h5, "r")

    def close(self):
        self.h5.close()

    def __enter__(self):
        self._idx = {gene.decode("utf-8"): i for i, gene in enumerate(self.h5["genes"][:])}
        self.genes = list(self._idx.keys())
        return self

    def load(self, genes: List[str]):
        idx = [self._idx[gene] for gene in genes]
        sorted_idx = sorted(idx)
        seqs = self.h5["seqs"][sorted_idx].astype(np.float32)
        attrs = self.h5["attrs"][sorted_idx].astype(np.float32)
        return seqs[idx], attrs[idx]

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
