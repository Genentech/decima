from pathlib import Path

import numpy as np
import torch
from grelu.sequence.format import convert_input_type

from decima.core.attribution import AttributionResult
from decima.constants import DECIMA_CONTEXT_SIZE

from conftest import attribution_h5_file, attribution_data


def test_AttributionResult(attribution_h5_file, attribution_data):
    with AttributionResult(str(attribution_h5_file), tss_distance=10_000, num_workers=1) as ar:
        assert len(ar.genes) == 10
        assert all(ar.genes == attribution_data['genes'])
        assert ar.model_name == 'v1_rep0'
        assert ar.genome == 'hg38'

        genes = attribution_data['genes']

        seqs, attrs = ar._load(str(attribution_h5_file), 0, 10_000, True)
        assert seqs.shape == (4, 20_000)
        assert attrs.shape == (4, 20_000)

        seqs, attrs = ar.load(genes, gene_mask=True)
        assert seqs.shape == (10, 5, 20_000)
        assert attrs.shape == (10, 4, 20_000)

        start, end = 163_840 - 10_000, 163_840 + 10_000
        assert np.allclose(convert_input_type(torch.from_numpy(seqs), "indices", input_type="one_hot"), attribution_data['sequences'][:, start:end])

        attribution = ar.load_attribution(genes[0])
        assert attribution.gene == genes[0]
        assert attribution.chrom == 'chr15'
        assert attribution.start == 43736410
        assert attribution.end == 43756410
        assert np.allclose(
            convert_input_type(torch.from_numpy(attribution.inputs), "indices", input_type="one_hot"),
            attribution_data['sequences'][0, start:end]
        )
        assert np.allclose(torch.from_numpy(attribution.attrs), attribution_data['attributions'][0][:, start:end])

    with AttributionResult(str(attribution_h5_file), tss_distance=None) as ar:

        seqs, attrs = ar.load(genes[:1])
        assert seqs.shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert attrs.shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert np.allclose(convert_input_type(torch.from_numpy(seqs), "indices", input_type="one_hot"), attribution_data['sequences'][:1])

        seqs, attrs = ar.load(genes[8:9])
        assert seqs.shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert attrs.shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert np.allclose(convert_input_type(torch.from_numpy(seqs), "indices", input_type="one_hot"), attribution_data['sequences'][8:9])

        seqs, attrs = ar.load(genes[:1], gene_mask=True)
        assert seqs.shape == (1, 5, DECIMA_CONTEXT_SIZE)
        assert attrs.shape == (1, 4, DECIMA_CONTEXT_SIZE)
        assert np.allclose(convert_input_type(torch.from_numpy(seqs[:, :4]), "indices", input_type="one_hot"), attribution_data['sequences'][:1])
        mask_idx = np.where(seqs[:, 4] == 1)[1]
        assert mask_idx[0] == 163_840
        assert mask_idx[-1] == 223_489

        attribution = ar.load_attribution(genes[0])
        assert attribution.gene == genes[0]
        assert attribution.chrom == 'chr15'
        assert attribution.start == 43582570
        assert attribution.end == 44106858
        assert np.allclose(convert_input_type(torch.from_numpy(attribution.inputs), "indices", input_type="one_hot"), attribution_data['sequences'][0])
        assert np.allclose(torch.from_numpy(attribution.attrs), attribution_data['attributions'][0])

    with AttributionResult([str(attribution_h5_file), str(attribution_h5_file)], tss_distance=10_000) as ar:
        assert len(ar.genes) == 10
        assert all(ar.genes == attribution_data['genes'])
        assert ar.model_name == ['v1_rep0', 'v1_rep0']
        assert ar.genome == 'hg38'

    with AttributionResult(str(attribution_h5_file), tss_distance=1_000_000) as ar:
        genes = attribution_data['genes']
        seqs, attrs = ar.load(genes)
        assert seqs.shape == (10, 4, 2_000_000)
        assert attrs.shape == (10, 4, 2_000_000)


def test_AttributionResult_recursive_seqlet_calling(attribution_h5_file, attribution_data):

    start, end = 163_840 - 10_000, 163_840 + 10_000
    df_peaks, df_motifs = AttributionResult._recursive_seqlet_calling(
        attribution_h5_file, 0, attribution_data['genes'][0], 10_000, 'chr1', start, end)
    assert df_peaks.shape == (17, 7)
    assert df_motifs.shape == (523, 11)

    with AttributionResult(str(attribution_h5_file), tss_distance=10_000, num_workers=1) as ar:
        df_peaks, df_motifs = ar.recursive_seqlet_calling(attribution_data['genes'][:4])
        assert df_peaks.shape == (69, 7)
        assert df_motifs.shape == (2288, 11)

    with AttributionResult([str(attribution_h5_file), str(attribution_h5_file)], tss_distance=10_000, num_workers=1, agg_func="mean") as ar:
        df_peaks, df_motifs = ar.recursive_seqlet_calling(attribution_data['genes'][:4])
        assert df_peaks.shape == (69, 7)
        assert df_motifs.shape == (2288, 11)
