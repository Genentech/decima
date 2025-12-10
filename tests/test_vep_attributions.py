import pytest
import h5py
from decima.constants import DECIMA_CONTEXT_SIZE
from decima.vep.attributions import variant_effect_attribution
from decima.core.attribution import VariantAttributionResult


@pytest.mark.long_running
def test_variant_effect_attribution():
    variant_effect_attribution(
        "tests/data/test.vcf",
        "tests/data/test.h5",
        model=0,
        method="inputxgradient",
    )
    with h5py.File("tests/data/test.h5", "r") as f:
        assert len(f['variants'][:]) == 82
        assert len(f['genes'][:]) == 82
        assert f['attribution'].shape == (82, 4, DECIMA_CONTEXT_SIZE)
        assert f['sequence'].shape == (82, DECIMA_CONTEXT_SIZE)
        assert f['attribution_alt'].shape == (82, 4, DECIMA_CONTEXT_SIZE)
        assert f['sequence_alt'].shape == (82, DECIMA_CONTEXT_SIZE)


def test_VariantAttributionResult(tmp_path, attribution_data):

    h5_path = tmp_path / "vep_test_attributions.h5"

    variants = ['chr1_1000018_G_A'] * len(attribution_data['genes'])

    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('genes', data=[name.encode('utf-8') for name in attribution_data['genes']])
        f.create_dataset('variants', data=[variant.encode('utf-8') for variant in variants])
        f.create_dataset('sequence', data=attribution_data['sequences'])
        f.create_dataset('attribution', data=attribution_data['attributions'])
        f.create_dataset('sequence_alt', data=attribution_data['sequences'])
        f.create_dataset('attribution_alt', data=attribution_data['attributions'])
        f.create_dataset('gene_mask_start', data=attribution_data['gene_mask_start'])
        f.create_dataset('gene_mask_end', data=attribution_data['gene_mask_end'])
        f.attrs['model_name'] = 'v1_rep0'
        f.attrs['genome'] = 'hg38'

    with VariantAttributionResult(str(h5_path), tss_distance=10_000, num_workers=1) as ar:
        genes = ar.genes
        seqs_ref, attrs_ref, seqs_alt, attrs_alt = ar.load(variants, genes)

        assert seqs_ref.shape == (len(attribution_data['genes']), 4, 20_000)
        assert attrs_ref.shape == (len(attribution_data['genes']), 4, 20_000)
        assert seqs_alt.shape == (len(attribution_data['genes']), 4, 20_000)
        assert attrs_alt.shape == (len(attribution_data['genes']), 4, 20_000)

        attribution_ref, attribution_alt = ar.load_attribution(variants[0], genes[0])
        assert attribution_ref.gene == genes[0]
        assert attribution_ref.chrom == 'chr15'
        assert attribution_ref.start == 43736410
        assert attribution_ref.end == 43756410

        assert attribution_alt.gene == genes[0]
        assert attribution_alt.chrom == 'chr15'
        assert attribution_alt.start == 43736410
        assert attribution_alt.end == 43756410
