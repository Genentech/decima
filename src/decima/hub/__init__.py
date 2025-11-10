import os
from typing import Union, Optional, List
import warnings
import wandb
from pathlib import Path
from tempfile import TemporaryDirectory
import anndata
from grelu.resources import get_artifact, DEFAULT_WANDB_HOST
from decima.constants import DEFAULT_ENSEMBLE, AVAILABLE_ENSEMBLES
from decima.model.lightning import LightningModel, EnsembleLightningModel


def login_wandb():
    """Login to wandb either as anonymous or as a user."""
    try:
        wandb.login(host=os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST), anonymous="never", timeout=0)
    except wandb.errors.UsageError:  # login anonymously if not logged in already
        wandb.login(host=os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST), relogin=True, anonymous="must", timeout=0)


def load_decima_model(model: Union[str, int, List[str]] = 0, device: Optional[str] = None):
    """Load a pre-trained Decima model from wandb or local path.

    Args:
        model: Model identifier or path. Can be:
            - int: Replicate number (0-3)
            - str: Model name on wandb
            - str: Path to local model checkpoint
            - List: list of local model checkpoints
        device: Device to load the model on. If None, automatically selects the best available device.

    Returns:
        LightningModel: A pre-trained Decima model instance loaded on the specified device.

    Raises:
        ValueError: If model identifier is invalid or not found.
    """
    if isinstance(model, LightningModel):
        return model

    elif model in AVAILABLE_ENSEMBLES:
        return EnsembleLightningModel([load_decima_model(i, device) for i in range(4)])

    elif isinstance(model, List):
        if len(model) == 1:
            return load_decima_model(model[0], device)
        else:
            return EnsembleLightningModel([load_decima_model(path, device) for path in model])

    elif model in {0, 1, 2, 3}:
        model_name = f"rep{model}"

    # Load directly from a path
    elif isinstance(model, str):
        if Path(model).exists():
            if model.endswith("ckpt"):
                return LightningModel.load_from_checkpoint(model, map_location=device)
            else:
                return LightningModel.load_safetensor(model, device=device)
        else:
            model_name = model

    else:
        raise ValueError(
            f"Invalid model: {model} it needs to be either a string of model_names on wandb, "
            "an integer of replicate number {0, 1, 2, 3}, a path to a local model or a list of paths."
        )

    # If left with a model name, load from environment/wandb
    if model_name.upper() in os.environ:
        if Path(os.environ[model_name.upper()]).exists():
            return LightningModel.load_safetensor(os.environ[model_name.upper()], device=device)
        else:
            warnings.warn(
                f"Model `{model_name}` provided in environment variables, "
                f"but not found in `{os.environ[model_name.upper()]}` "
                f"Trying to download `{model_name}` from wandb."
            )

    art = get_artifact(model_name, project="decima")
    with TemporaryDirectory() as d:
        art.download(d)
        return LightningModel.load_safetensor(Path(d) / f"{model_name}.safetensors", device=device)


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
        if Path(os.environ["DECIMA_METADATA"]).exists():
            return anndata.read_h5ad(os.environ["DECIMA_METADATA"])
        else:
            warnings.warn(
                f"Metadata `{os.environ['DECIMA_METADATA']}` provided in environment variables, "
                f"but not found in `{os.environ['DECIMA_METADATA']}` "
                f"Trying to download `metadata` from wandb."
            )

    art = get_artifact("metadata", project="decima")
    with TemporaryDirectory() as d:
        art.download(d)
        return anndata.read_h5ad(Path(d) / "metadata.h5ad")
