import os
import json
from typing import Union, Optional, List
import warnings
import wandb
from pathlib import Path
from tempfile import TemporaryDirectory
import anndata
from grelu.resources import get_artifact, DEFAULT_WANDB_HOST
from decima.constants import DEFAULT_ENSEMBLE, ENSEMBLE_MODELS, MODEL_METADATA
from decima.model.lightning import LightningModel, EnsembleLightningModel


def login_wandb():
    """Login to wandb either as anonymous or as a user."""
    host = os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST)
    try:
        wandb.login(host=host, anonymous="never", timeout=0)
    except wandb.errors.UsageError:  # login anonymously if not logged in already
        wandb.login(host=host, relogin=True, anonymous="must", timeout=0)


def load_decima_model(model: Union[str, int, List[str]] = DEFAULT_ENSEMBLE, device: Optional[str] = None):
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

    elif model in ENSEMBLE_MODELS:
        return EnsembleLightningModel(
            [load_decima_model(model_name, device) for model_name in MODEL_METADATA[model]],
            name=model,
        )

    elif isinstance(model, List):
        if len(model) == 1:
            return load_decima_model(model[0], device)
        else:
            return EnsembleLightningModel([load_decima_model(path, device) for path in model], name=model)

    # Load directly from a path
    if model in MODEL_METADATA:
        model_name = MODEL_METADATA[model]["name"]
        if "model_path" in MODEL_METADATA[model]:  # if model path exist in metadata load it from the path
            return load_decima_model(MODEL_METADATA[model]["model_path"], device)
    elif isinstance(model, str) and Path(model).exists():
        if model.endswith("ckpt"):
            return LightningModel.load_from_checkpoint(model, map_location=device)
        else:
            return LightningModel.load_safetensor(model, device=device)
    else:
        raise ValueError(
            f"Invalid model: {model} it needs to be either a string of model_names on wandb ("
            f"{list(MODEL_METADATA.keys())}), path to a local model, or a list of paths."
        )
    # load model from wandb
    art = get_artifact(
        model_name,
        project="decima",
        host=os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST),
    )
    with TemporaryDirectory() as d:
        art.download(d)
        return LightningModel.load_safetensor(Path(d) / f"{model_name}.safetensors", device=device)


def load_decima_metadata(name_or_path: Optional[str] = None):
    """Load the Decima metadata from wandb.

    Args:
        name_or_path: Path to local metadata file or name of the model to load metadata for using wandb. If None, default model's metadata will be downloaded from wandb.

    Returns:
        An AnnData object containing the Decima metadata.
    """
    if name_or_path is not None:
        if Path(name_or_path).exists():
            return anndata.read_h5ad(name_or_path)

    name_or_path = name_or_path or DEFAULT_ENSEMBLE

    if name_or_path in ENSEMBLE_MODELS:
        name_or_path = MODEL_METADATA[name_or_path][0]

    if name_or_path in MODEL_METADATA:
        metadata = MODEL_METADATA[name_or_path]

    if "metadata_path" in metadata:
        if Path(metadata["metadata_path"]).exists():
            return anndata.read_h5ad(metadata["metadata_path"])
        else:
            warnings.warn(
                f"Metadata `{metadata['metadata_path']}` provided in environment variables, "
                f"but not found in `{metadata['metadata_path']}` "
                f"Trying to download `metadata` from wandb."
            )

    art = get_artifact(
        metadata["metadata"],
        project="decima",
        host=os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST),
    )
    with TemporaryDirectory() as d:
        art.download(d)
        return anndata.read_h5ad(Path(d) / f"{metadata['metadata']}.h5ad")
