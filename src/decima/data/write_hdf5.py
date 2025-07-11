import h5py
import numpy as np
from grelu.sequence.format import convert_input_type
from grelu.sequence.utils import get_unique_length


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
        print("Writing task indices")
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
        arr = convert_input_type(arr, "indices", genome="hg38")
        print(f"Writing sequence array of shape: {arr.shape}")
        f.create_dataset("sequences", shape=arr.shape, dtype=np.int8, data=arr)

    print("Done!")
