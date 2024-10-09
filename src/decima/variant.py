from grelu.sequence.utils import reverse_complement, get_unique_length


def process_variants(variants, ad, min_from_end=0):

    # Match to gene intervals
    orig_len = len(variants)
    variants = variants[variants.gene.isin(ad.var_names)]
    print(f'dropped {orig_len - len(variants)} variants because the gene was not found in ad.var')
    variants = variants.merge(ad.var[['start', 'end', 'strand', 'gene_mask_start']], left_on='gene', right_index=True, how='left')

    # Get relative position
    variants['rel_pos'] = variants.apply(lambda row: row.pos - row.start if row.strand == "+" else row.end - row.pos, axis=1)

    # Filter by relative position
    orig_len = len(variants)
    interval_len = get_unique_length(ad.var)
    variants = variants[(variants.rel_pos > min_from_end) & (variants.rel_pos < interval_len - min_from_end)]
    print(f'dropped {orig_len - len(variants)} variants because the variant did not fit in the interval')
    
    # Reverse complement the alleles for - strand genes
    variants['ref_tx'] = variants.apply(lambda row: row.ref if row.strand=="+" else reverse_complement(row.ref), axis=1)
    variants['alt_tx'] = variants.apply(lambda row: row.alt if row.strand=="+" else reverse_complement(row.alt), axis=1)

    # Get distance from TSS
    variants['tss_dist'] = variants.rel_pos - variants.gene_mask_start

    return variants
