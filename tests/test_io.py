from decima.utils.io import read_fasta_gene_mask


def test_read_fasta():
    df = read_fasta_gene_mask("tests/data/seqs.fasta")
    assert df.columns.tolist() == ['seq', 'gene_mask_start', 'gene_mask_end']
    assert df.index.tolist() == ['CD68', 'SPI1']
