import pytest
from decima.utils.inject import SeqBuilder, prepare_seq_alt_allele


def test_seq_builder():
    # 0 based start and 1 based end
    builder = SeqBuilder(chrom="chr1", start=100, end=200, anchor=150)
    builder.inject({"chrom": "chr1", "pos": 120, "ref": "A", "alt": "C"})
    builder.inject({"chrom": "chr1", "pos": 180, "ref": "G", "alt": "T"})

    fragments = list(builder._construct())
    assert len(fragments) == 5

    assert len(fragments[0]) == 19
    assert fragments[1] == "C"
    assert len(fragments[2]) == 59
    assert fragments[3] == "T"
    assert len(fragments[4]) == 20

    assert len(builder.concat()) == 100


def test_seq_builder_indel():
    builder = SeqBuilder(chrom="chr1", start=100, end=200, anchor=150)
    builder.inject({"chrom": "chr1", "pos": 120, "ref": "A", "alt": "CC"})
    builder.inject({"chrom": "chr1", "pos": 180, "ref": "C", "alt": "TT"})

    assert builder.start_shift == 1
    assert builder.end_shift == -1

    fragments = list(builder._construct())
    assert len(fragments) == 5

    assert len(fragments[0]) == 18
    assert fragments[1] == "CC"
    assert len(fragments[2]) == 59
    assert fragments[3] == "TT"
    assert len(fragments[4]) == 19

    assert len(builder.concat()) == 100


def test_seq_builder_split_variant():
    l_variant, r_variant = SeqBuilder._split_variant({"chrom": "chr1", "pos": 148, "ref": "CCCCC", "alt": "T"}, pos=150)
    assert l_variant == {"chrom": "chr1", "pos": 148, "ref": "CC", "alt": "T"}
    assert r_variant == {"chrom": "chr1", "pos": 150, "ref": "CCC", "alt": ""}


def test_seq_builder_indel_with_anchor():
    builder = SeqBuilder(chrom="chr1", start=100, end=200, anchor=150, track=[110, 149, 150, 181])
    builder.inject({"chrom": "chr1", "pos": 120, "ref": "A", "alt": "CC"})
    builder.inject({"chrom": "chr1", "pos": 180, "ref": "C", "alt": "TT"})
    builder.inject({"chrom": "chr1", "pos": 148, "ref": "CCCCC", "alt": "T"})

    assert builder.start_shift == 0
    assert builder.end_shift == 2

    fragments = list(builder._construct())

    assert len(fragments) == 9

    assert len(fragments[0]) == 19
    assert fragments[1] == "CC"
    assert len(fragments[2]) == 27
    assert fragments[3] == "T"

    assert len(fragments[6]) == 27
    assert fragments[7] == "TT"
    assert len(fragments[8]) == 22

    assert len(builder.concat()) == 100

    assert builder.shifts == {110: 0, 149: 1, 181: -2, 150: 0}


def test_seq_builder_redundant_variants():
    builder = SeqBuilder(chrom="chr1", start=100, end=200, anchor=150)
    builder.inject({"chrom": "chr1", "pos": 120, "ref": "A", "alt": "CC"})

    with pytest.raises(ValueError):
        builder.inject({"chrom": "chr1", "pos": 120, "ref": "C", "alt": "TT"})
