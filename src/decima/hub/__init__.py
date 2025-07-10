import os
from typing import Union, Optional
import wandb
from pathlib import Path
from tempfile import TemporaryDirectory
import anndata
from grelu.resources import get_artifact, DEFAULT_WANDB_HOST
from decima.model.lightning import LightningModel, EnsembleLightningModel


def login_wandb():
    try:
        wandb.login(host=os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST), anonymous="never", timeout=0)
    except wandb.errors.UsageError:  # login anonymously if not logged in already
        wandb.login(host=os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST), relogin=True, anonymous="must", timeout=0)


def get_model_name(model: Union[str, int] = 0) -> str:
    if isinstance(model, int):
        return f"decima_rep{model}"
    elif isinstance(model, str):
        return model
    else:
        raise ValueError(
            f"Invalid model: {model} it need to be a string of model_name on wandb or an integer of replicate number {0, 1, 2, 3}"
        )


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
    elif model == "ensemble":
        model = EnsembleLightningModel(
            [
                load_decima_model(0, device),
                load_decima_model(1, device),
                load_decima_model(2, device),
                load_decima_model(3, device),
            ]
        )
        model.name = "ensemble"
        return model
    elif isinstance(model, str):
        model_name = get_model_name(model)
        if Path(model).exists():
            model = LightningModel.load_from_checkpoint(model, map_location=device)
            model.name = model_name
            return model
    elif isinstance(model, int):
        model_name = get_model_name(model)
    else:
        raise ValueError(
            f"Invalid model: {model} it need to be a string of model_name on wandb "
            "or an integer of replicate number {0, 1, 2, 3}, or a path to a local model"
        )

    if model_name.upper() in os.environ:
        return LightningModel.load_from_checkpoint(os.environ[model_name.upper()], map_location=device)

    art = get_artifact(model_name, project="decima")
    with TemporaryDirectory() as d:
        art.download(d)
        model = LightningModel.load_from_checkpoint(Path(d) / "model.ckpt", map_location=device)
        model.name = str(model_name)
        return model


def load_decima_metadata(path: Optional[str] = None):
    """Load the Decima metadata from wandb.

    Args:
        path: Path to local metadata file. If None, downloads from wandb.

    Returns:
        An AnnData object containing the Decima metadata.
    """
    if path is not None:
        return anndata.read_h5ad(path)

    if "DECIMA_METADATA" in os.environ:
        return anndata.read_h5ad(os.environ["DECIMA_METADATA"])

    art = get_artifact("decima_metadata", project="decima")
    with TemporaryDirectory() as d:
        art.download(d)
        return anndata.read_h5ad(Path(d) / "metadata.h5ad")
