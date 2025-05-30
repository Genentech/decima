import anndata
from grelu.sequence.utils import get_unique_length, reverse_complement
from decima.core import DecimaResult


def process_variants(variants, ad=None, min_from_end=0):
    raise DeprecationWarning("process_variants is deprecated. Use `VariantDataset.overlap_genes` instead.")
    # Match to gene intervals

    if ad is None:
        result = DecimaResult.load()
    elif isinstance(ad, str):
        result = DecimaResult.load(ad)
    elif isinstance(ad, anndata.AnnData):
        result = DecimaResult(ad)
    else:
        raise ValueError(f"Invalid ad: {ad} (must be None, str, or anndata.AnnData)")

    # TODO: overlap with gene intervals
    orig_len = len(variants)
    variants = variants[variants.gene.isin(result.genes)]
    print(f"dropped {orig_len - len(variants)} variants because the gene was not found in ad.var")

    variants = variants.merge(
        ad.var[["start", "end", "strand", "gene_mask_start"]],
        left_on="gene",
        right_index=True,
        how="left",
    )

    # Get relative position
    variants["rel_pos"] = variants.apply(
        lambda row: row.pos - row.start if row.strand == "+" else row.end - row.pos,
        axis=1,
    )

    # Filter by relative position
    orig_len = len(variants)
    interval_len = get_unique_length(result.anndata.var)
    variants = variants[(variants.rel_pos > min_from_end) & (variants.rel_pos < interval_len - min_from_end)]
    print(f"dropped {orig_len - len(variants)} variants because the variant did not fit in the interval")

    # Reverse complement the alleles for - strand genes
    variants["ref_tx"] = variants.apply(
        lambda row: row.ref if row.strand == "+" else reverse_complement(row.ref),
        axis=1,
    )
    variants["alt_tx"] = variants.apply(
        lambda row: row.alt if row.strand == "+" else reverse_complement(row.alt),
        axis=1,
    )

    # Get distance from TSS
    variants["tss_dist"] = variants.rel_pos - variants.gene_mask_start

    return variants
