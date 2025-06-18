from typing import Dict, List, Generator
import warnings
from collections import Counter
from grelu.sequence.utils import reverse_complement
from grelu.sequence.format import intervals_to_strings

from decima.core.metadata import GeneMetadata


class SeqBuilder:
    """
    Build the sequence from the variants.

    Args:
        chrom: chromosome
        start: start position
        end: end position
        anchor: anchor position
        track: track positions shifts due to indels.
    """

    def __init__(self, chrom: str, start: int, end: int, anchor: int, track: List[int] = None):
        self.chrom = chrom
        self.start = start
        self.end = end
        self.variants = list()
        self.anchor = anchor
        self.start_shift = 0  # how much interval is shifted to the left upstream
        self.end_shift = 0  # how much interval is shifted to the right downstream
        self.shifts = {pos: 0 for pos in track or list()}

    @staticmethod
    def _split_variant(variant, pos):
        start = variant["pos"]
        end = start + len(variant["ref"])

        assert start <= pos < end, f"Variant position {variant['pos']} is out of range [{start}, {end})"
        _pos = pos - start

        left_variant = {
            "chrom": variant["chrom"],
            "pos": start,
            "ref": variant["ref"][:_pos],
            "alt": variant["alt"][:_pos],
        }
        right_variant = {
            "chrom": variant["chrom"],
            "pos": pos,
            "ref": variant["ref"][_pos:],
            "alt": variant["alt"][_pos:],
        }
        return left_variant, right_variant

    def inject(self, variant: Dict):
        """
        Inject the variant into the sequence.

        Args:
            variant: variant to inject in the format of {"chrom": str, "pos": int, "ref": str, "alt": str}

        Returns:
            self
        """
        variant = dict(variant)
        variant["ref"] = variant["ref"].replace(".", "")
        variant["alt"] = variant["alt"].replace(".", "")

        if variant["chrom"] != self.chrom:
            warnings.warn(f"Variant chromosome `{variant['chrom']}` does not match `{self.chrom}`. Skipping...")
            return self

        for i in self.variants:
            if variant["pos"] == i["pos"]:
                raise ValueError(
                    f"At this position `{variant['pos']}` there is already a variant `{i}` "
                    f"thus cannot inject `{variant}` variant at the same position."
                    " Please check variant and ensure not redundant positions."
                )

        variant_start = variant["pos"]
        variant_end = variant_start + len(variant["ref"])

        if variant_start < self.start:
            warnings.warn(
                f"Variant position `{variant['pos']}` is upstream of the interval `[{self.start}, {self.end}]`. Skipping..."
            )
            return self
        elif self.end < variant_end:
            warnings.warn(
                f"Variant position `{variant['pos']}` is downstream of the interval `[{self.start}, {self.end}]`. Skipping..."
            )
            return self

        if variant_start < self.anchor < variant_end:
            left_variant, right_variant = self._split_variant(variant, self.anchor)
            self.inject(left_variant)
            self.inject(right_variant)
            return self

        self.variants.append(variant)

        diff = len(variant["ref"]) - len(variant["alt"])
        if variant["pos"] < self.anchor:
            self.start_shift -= diff
        else:
            self.end_shift += diff

        for pos in self.shifts:
            if variant_start < pos < variant_end:
                l_variant, r_variant = self._split_variant(variant, pos)

                if self.anchor < pos:
                    _variant_start = l_variant["pos"]
                    _variant_end = l_variant["pos"] + len(l_variant["ref"])
                    _diff = len(l_variant["ref"]) - len(l_variant["alt"])
                else:
                    _variant_start = r_variant["pos"]
                    _variant_end = r_variant["pos"] + len(r_variant["ref"])
                    _diff = len(r_variant["ref"]) - len(r_variant["alt"])
            else:
                _variant_start = variant_start
                _variant_end = variant_end
                _diff = diff

            if self.anchor <= _variant_start < pos:
                self.shifts[pos] -= _diff
            elif pos < _variant_end <= self.anchor:
                self.shifts[pos] += _diff

        return self

    def _construct(self) -> Generator[str, None, None]:
        """
        Construct the sequence from the variants.

        Returns:
            Generator[str, None, None]: the sequence.
        """
        start = self.start + self.start_shift
        end = self.end + self.end_shift

        seq = intervals_to_strings({"chrom": self.chrom, "start": start, "end": end}, genome="hg38")
        start += 1  # 0 based to 1 based start

        variants = sorted(self.variants, key=lambda x: x["pos"])

        variant_end = 0
        for i, variant in enumerate(variants):
            if i == 0:
                prev_end = 0
            else:
                prev_variant = variants[i - 1]
                prev_end = prev_variant["pos"] + len(prev_variant["ref"]) - start

            variant_start = variant["pos"] - start
            variant_end = variant_start + len(variant["ref"])

            yield seq[prev_end:variant_start]
            yield variant["alt"]

        yield seq[variant_end:]

    def concat(self) -> str:
        """
        Build the string from sequence objects.

        Returns:
          str: the final sequence.
        """
        return "".join(self._construct())


def prepare_seq_alt_allele(gene: GeneMetadata, variants: List[Dict]):
    """
    Prepare the sequence and alt allele for a gene.

    Example:
        --------------{---------}--------: ref
        *------x------{---------}--------: alt new sequence fetched from the upsteam due to deletion.

        --------------{---------}--------: ref
        --------------{----++---}----++--: alt 4 bp cropped from the downstream due to insertion.
                      ^anchor

    Args:
        gene: gene metadata in the format of GeneMetadata.
        variants: variants to inject in the format of [{"chrom": str, "pos": int, "ref": str, "alt": str}, ...].

    Returns:
        tuple: the sequence (str) and gene mask start and end positions (int, int)

    """
    count_positions = Counter(variant["pos"] for variant in variants)
    if len(count_positions) != len(variants):
        duplicates = [pos for pos, count in count_positions.items() if count > 1]
        raise ValueError(
            f"Two variants cannot have the same position but got redundant variants at positions: {duplicates}"
        )

    anchor = gene.gene_end if gene.strand == "-" else gene.gene_start

    builder = SeqBuilder(
        chrom=gene.chrom, start=gene.start, end=gene.end, anchor=anchor, track=[gene.gene_start, gene.gene_end]
    )
    for variant in variants:
        builder.inject(variant)

    seq = builder.concat()
    gene_start_shift = builder.shifts[gene.gene_start]
    gene_end_shift = builder.shifts[gene.gene_end]

    if gene.strand == "-":
        assert gene_end_shift == 0
        seq = reverse_complement(seq, input_type="strings")
        gene_mask = (gene.gene_mask_start - gene_start_shift, gene.gene_mask_end - gene_end_shift)
    else:
        assert gene_start_shift == 0
        gene_mask = (gene.gene_mask_start + gene_start_shift, gene.gene_mask_end + gene_end_shift)

    return seq, gene_mask
