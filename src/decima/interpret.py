import os
import sys

import numpy as np
import pandas as pd
import torch
from captum.attr import InputXGradient
from grelu.interpret.motifs import scan_sequences
from grelu.sequence.format import convert_input_type
from grelu.transforms.prediction_transforms import Aggregate, Specificity
from scipy.signal import find_peaks

src_dir = f"{os.path.dirname(__file__)}/../src/decima/"
sys.path.append(src_dir)
from read_hdf5 import extract_gene_data


def attributions(
    gene,
    tasks,
    model,
    device,
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

    tss_pos = np.where(inputs[-1] == 1)[0][0]
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
    device = torch.device(device)

    attributer = method(model.to(device))
    with torch.no_grad():
        attr = attributer.attribute(inputs.to(device), **kwargs).cpu().numpy()[:4]

    model.reset_transform()
    return attr, tss_pos


def find_attr_peaks(attr, tss_pos=None, n=5, min_dist=6):
    peaks, heights = find_peaks(attr.sum(0), height=0.1, distance=min_dist)
    peaks = pd.DataFrame({"peak": peaks, "height": heights["peak_heights"]})
    if tss_pos is not None:
        peaks["from_tss"] = peaks["peak"] - tss_pos
    peaks = peaks.sort_values("height", ascending=False).head(n)
    return peaks.reset_index(drop=True)


def scan_attributions(seq, attr, motifs, peaks, names=None, pthresh=1e-3, rc=True, window=18):
    # Attributions and sequences
    peak_attrs = np.stack([attr[:, peak - window // 2 : peak + window // 2] for peak in peaks.peak])
    peak_seqs = torch.stack([seq[:, peak - window // 2 : peak + window // 2] for peak in peaks.peak])

    # Scan
    results = scan_sequences(
        seqs=convert_input_type(peak_seqs, "strings"),
        motifs=motifs,
        names=names,
        pthresh=pthresh,
        rc=rc,
        attrs=peak_attrs,
    )
    results.sequence = results.sequence.astype(int)
    return results.merge(peaks.reset_index(drop=True), left_on="sequence", right_index=True)
