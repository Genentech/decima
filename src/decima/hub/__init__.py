from typing import Union, Optional
import wandb
from pathlib import Path
from tempfile import TemporaryDirectory
import anndata
from grelu.resources import get_artifact, DEFAULT_WANDB_HOST
from decima.model.lightning import LightningModel


def login_wandb():
    try:
        wandb.login(host=DEFAULT_WANDB_HOST, anonymous="never", timeout=0)
    except wandb.errors.UsageError:  # login anonymously if not logged in already
        wandb.login(host=DEFAULT_WANDB_HOST, relogin=True, anonymous="must", timeout=0)


def load_decima_model(model: Union[str, int] = 0, device: Optional[str] = None):
    """Load a pre-trained Decima model from wandb or local path.

    Args:
        model: Model identifier or path. Can be:
            - int: Replicate number (0-3)
            - str: Model name on wandb
            - str: Path to local model checkpoint
        device: Device to load the model on. If None, automatically selects the best available device.

    Returns:
        LightningModel: A pre-trained Decima model instance loaded on the specified device.

    Raises:
        ValueError: If model identifier is invalid or not found.
    """
    if isinstance(model, LightningModel):
        return model
    elif isinstance(model, str):
        if Path(model).exists():
            return LightningModel.load_from_checkpoint(model, map_location=device)
        model_name = model
    elif isinstance(model, int):
        model_name = f"decima_rep{model}"
    else:
        raise ValueError(
            f"Invalid model: {model} it need to be a string of model_name on wandb "
            "or an integer of replicate number {0, 1, 2, 3}, or a path to a local model"
        )

    art = get_artifact(model_name, project="decima")
    with TemporaryDirectory() as d:
        art.download(d)
        return LightningModel.load_from_checkpoint(Path(d) / "model.ckpt", map_location=device)


def load_decima_metadata(path: Optional[str] = None):
    """Load the Decima metadata from wandb.

    Args:
        path: Path to local metadata file. If None, downloads from wandb.

    Returns:
        An AnnData object containing the Decima metadata.
    """
    if path is not None:
        return anndata.read_h5ad(path)

    art = get_artifact("decima_metadata", project="decima")
    with TemporaryDirectory() as d:
        art.download(d)
        return anndata.read_h5ad(Path(d) / "metadata.h5ad")
