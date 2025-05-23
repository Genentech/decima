from decima.utils.io import read_fasta


def test_read_fasta():
    df = read_fasta("tests/data/test.fasta")
    assert df.columns == ['seq', 'gene_mask_start', 'gene_mask_end']
