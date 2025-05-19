from pathlib import Path
from tempfile import TemporaryDirectory
import anndata
from grelu.resources import get_artifact
from decima.model.lightning import LightningModel


def load_decima_model(rep: int = 0, device: str = "cpu"):
    """Load a pre-trained Decima model from wandb.

    Args:
        rep: Replicate number of the model to load (default: 0)
        device: Device to load the model on (default: "cpu")

    Returns:
        A pre-trained LightningModel instance
    """
    art = get_artifact(f"decima_rep{rep}", project="decima")
    with TemporaryDirectory() as d:
        art.download(d)
        return LightningModel.load_from_checkpoint(Path(d) / "model.ckpt", map_location=device)


def load_decima_metadata():
    """Load the Decima metadata from wandb.

    Returns:
        An AnnData object containing the Decima metadata.
    """
    art = get_artifact("decima_metadata", project="decima")
    with TemporaryDirectory() as d:
        art.download(d)
        return anndata.read_h5ad(Path(d) / "metadata.h5ad")
