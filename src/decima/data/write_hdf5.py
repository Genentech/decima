import h5py
import numpy as np
from grelu.sequence.format import strings_to_indices
from grelu.sequence.utils import get_unique_length
from grelu.io.genome import read_sizes, get_genome


def _get_padded_seq(chrom, start, end, strand, sizes, genome):
    target_len = end - start
    
    if start < 0:
        seq = str(genome.get_seq(chrom, 1, end, rc=strand == "-")).upper()
        seq = 'N' * (target_len - len(seq)) + seq

    else:
        chr_end = sizes[sizes.chrom==chrom].size
        if end > chr_end:
            seq = str(genome.get_seq(chrom, start+1, chr_end, rc=strand == "-")).upper()
            seq = 'N' * (target_len - len(seq)) + seq

        else:
            seq = str(genome.get_seq(chrom, start, end, rc=strand == "-")).upper()

    assert len(seq) == target_len
    return seq
        

def intervals_to_strings_N_pad(intervals, genome = 'hg38'):
    """
    Extract DNA sequences from the specified intervals in a genome.

    Args:
        intervals: A pandas DataFrame, Series or dictionary containing
            the genomic interval(s) to extract.
        genome: Name of the genome to use.

    Returns:
        A list of DNA sequences extracted from the intervals.
    """
    # Get chromosome sizes
    sizes = read_sizes(genome)
    
    # Get genome
    genome = get_genome(genome)

    # Extract sequence for a single interval
    seqs = intervals.apply(
                lambda row:  _get_seq_N_pad(
                    row["chrom"], row["start"], row["end"], row["strand"], sizes, genome
                ), axis=1,
            ).tolist()

    assert len(seqs) == len(intervals)
    return seqs


def write_hdf5(file, ad, pad=0):
    # Calculate seq_len
    seq_len = get_unique_length(ad.var)

    with h5py.File(file, "w") as f:
        # Metadata
        print("Writing metadata")
        f.create_dataset("pad", shape=(), data=pad)
        f.create_dataset("seq_len", shape=(), data=seq_len)
        padded_seq_len = seq_len + 2 * pad
        f.create_dataset("padded_seq_len", shape=(), data=padded_seq_len)

        # Tasks
        print("Writing tasks")
        tasks = np.array(ad.obs.index)
        f.create_dataset("tasks", shape=tasks.shape, data=tasks)

        # Genes
        arr = np.array(ad.var[["dataset"]].reset_index())
        print(f"Writing genes array of shape: {arr.shape}")
        f.create_dataset("genes", shape=arr.shape, data=arr)

        # Labels
        arr = np.expand_dims(ad.X.T.astype(np.float32), 2)
        print(f"Writing labels array of shape: {arr.shape}")
        f.create_dataset("labels", shape=arr.shape, dtype=np.float32, data=arr)

        # Gene masks
        print("Making gene masks")
        shape = (ad.var.shape[0], padded_seq_len)
        arr = np.zeros(shape=shape)
        for i, row in enumerate(ad.var.itertuples()):
            arr[i, row.gene_mask_start + pad : row.gene_mask_end + pad] += 1
        print(f"Writing mask array of shape: {arr.shape}")
        f.create_dataset("masks", shape=shape, dtype=np.float32, data=arr)

        # Sequences
        print("Encoding sequences")
        arr = ad.var[["chrom", "start", "end", "strand"]].copy()
        arr.start = arr.start - pad
        arr.end = arr.end + pad
        arr = intervals_to_strings_N_pad(arr, genome="hg38")
        arr = strings_to_indices(arr)
        print(f"Writing sequence array of shape: {arr.shape}")
        f.create_dataset("sequences", shape=arr.shape, dtype=np.int8, data=arr)

    print("Done!")
