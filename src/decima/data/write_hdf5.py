import h5py
import numpy as np
from grelu.io.genome import get_genome
from grelu.sequence.format import _BASE_LUT
from grelu.sequence.utils import get_unique_length
from tqdm import tqdm


def write_hdf5(file, ad, pad=0, genome="hg38", batch_size=1000):
    """Write AnnData object to HDF5 file.

    Args:
        file: Path to the HDF5 file to write
        ad: AnnData object containing the data
        pad: Amount of padding to add. Defaults to 0
        genome: Genome name or path to the genome fasta file. Defaults to "hg38"
        batch_size: Number of genes per write batch. Defaults to 1000
    """
    seq_len = get_unique_length(ad.var)
    padded_seq_len = seq_len + 2 * pad
    n_genes = ad.var.shape[0]
    genome_obj = get_genome(genome)

    intervals = ad.var[["chrom", "start", "end", "strand"]].copy()
    intervals["start"] = intervals["start"] - pad
    intervals["end"] = intervals["end"] + pad

    with h5py.File(file, "w") as f:
        # Metadata
        print("Writing metadata")
        f.create_dataset("pad", shape=(), data=pad)
        f.create_dataset("seq_len", shape=(), data=seq_len)
        f.create_dataset("padded_seq_len", shape=(), data=padded_seq_len)

        # Tasks
        print("Writing task indices")
        tasks = np.array(ad.obs.index)
        f.create_dataset("tasks", shape=tasks.shape, data=tasks)

        # Genes
        arr = np.array(ad.var[["dataset"]].reset_index())
        print(f"Writing genes array of shape: {arr.shape}")
        f.create_dataset("genes", shape=arr.shape, data=arr)

        # Labels
        print("Writing labels")
        X = ad.X.toarray() if hasattr(ad.X, "toarray") else np.asarray(ad.X)
        arr = np.expand_dims(X.T.astype(np.float32), 2)
        print(f"  shape: {arr.shape}")
        f.create_dataset("labels", shape=arr.shape, dtype=np.float32, data=arr)
        del X, arr

        # Masks and sequences — written in batches to avoid OOM
        print("Writing masks and sequences")
        masks_ds = f.create_dataset(
            "masks", shape=(n_genes, padded_seq_len), dtype=np.float32
        )
        seqs_ds = f.create_dataset(
            "sequences", shape=(n_genes, padded_seq_len), dtype=np.int8
        )

        n_batches = (n_genes + batch_size - 1) // batch_size
        for b in tqdm(range(n_batches), desc="Batches"):
            start_i = b * batch_size
            end_i = min(start_i + batch_size, n_genes)
            batch_var = ad.var.iloc[start_i:end_i]
            batch_iv = intervals.iloc[start_i:end_i]

            masks = np.zeros((end_i - start_i, padded_seq_len), dtype=np.float32)
            seqs = np.empty((end_i - start_i, padded_seq_len), dtype=np.int8)

            for j, (row_var, row_iv) in enumerate(
                zip(batch_var.itertuples(), batch_iv.itertuples())
            ):
                masks[j, row_var.gene_mask_start + pad : row_var.gene_mask_end + pad] = 1.0
                seq = str(
                    genome_obj.get_seq(
                        row_iv.chrom,
                        row_iv.start + 1,
                        row_iv.end,
                        rc=row_iv.strand == "-",
                    )
                ).upper()
                seqs[j] = _BASE_LUT[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]

            masks_ds[start_i:end_i] = masks
            seqs_ds[start_i:end_i] = seqs

    print("Done!")
