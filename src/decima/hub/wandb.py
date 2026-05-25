"""Wandb-based model loading for internal/private model access."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union, Optional, List
import warnings
import wandb
import anndata
from decima.constants import DEFAULT_ENSEMBLE, ENSEMBLE_MODELS, MODEL_METADATA
from decima.model.lightning import LightningModel, EnsembleLightningModel

WANDB_ENTITY = "grelu"
WANDB_PROJECT = "decima"
DEFAULT_WANDB_HOST = "https://wandb.ai"


def _get_host(host: Optional[str] = None) -> str:
    return host or os.environ.get("WANDB_HOST", DEFAULT_WANDB_HOST)


def login_wandb(host: Optional[str] = None):
    """Login to wandb either as anonymous or as a user."""
    h = _get_host(host)
    try:
        wandb.login(host=h, anonymous="never", timeout=0)
    except wandb.errors.UsageError:
        wandb.login(host=h, relogin=True, anonymous="must", timeout=0)


def _get_artifact(name: str, host: Optional[str] = None):
    api = wandb.Api(overrides={"host": _get_host(host)})
    return api.artifact(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{name}:latest")


def load_decima_model(
    model: Union[str, int, List[str]] = DEFAULT_ENSEMBLE,
    device: Optional[str] = None,
    host: Optional[str] = None,
):
    """Load a pre-trained Decima model from wandb or a local path."""
    if isinstance(model, LightningModel):
        return model

    if model in ENSEMBLE_MODELS:
        return EnsembleLightningModel(
            [load_decima_model(m, device, host) for m in MODEL_METADATA[model]],
            name=model,
        )

    if isinstance(model, list):
        if len(model) == 1:
            return load_decima_model(model[0], device, host)
        return EnsembleLightningModel([load_decima_model(p, device, host) for p in model], name=model)

    if isinstance(model, str) and Path(model).exists():
        if model.endswith(".ckpt"):
            return LightningModel.load_from_checkpoint(model, map_location=device)
        return LightningModel.load_safetensor(model, device=device)

    if model in MODEL_METADATA:
        if "model_path" in MODEL_METADATA[model]:
            return load_decima_model(MODEL_METADATA[model]["model_path"], device, host)
        name = MODEL_METADATA[model]["name"]
        art = _get_artifact(name, host)
        with TemporaryDirectory() as d:
            art.download(d)
            return LightningModel.load_safetensor(Path(d) / f"{name}.safetensors", device=device)

    raise ValueError(
        f"Invalid model: {model}. Must be a known model name {list(MODEL_METADATA.keys())}, "
        "a local path, or a list of local paths."
    )


def load_decima_metadata(name_or_path: Optional[str] = None, host: Optional[str] = None):
    """Load the Decima metadata from wandb or a local path."""
    if name_or_path is not None and Path(name_or_path).exists():
        return anndata.read_h5ad(name_or_path)

    name_or_path = name_or_path or DEFAULT_ENSEMBLE

    if name_or_path in ENSEMBLE_MODELS:
        name_or_path = MODEL_METADATA[name_or_path][0]

    if name_or_path in MODEL_METADATA:
        metadata = MODEL_METADATA[name_or_path]
        if "metadata_path" in metadata:
            if Path(metadata["metadata_path"]).exists():
                return anndata.read_h5ad(metadata["metadata_path"])
            warnings.warn(
                f"Metadata path `{metadata['metadata_path']}` not found. Downloading from wandb."
            )
        art = _get_artifact(metadata["metadata"], host)
        with TemporaryDirectory() as d:
            art.download(d)
            return anndata.read_h5ad(Path(d) / f"{metadata['metadata']}.h5ad")

    raise ValueError(f"Unknown model: {name_or_path}")


def download_decima_weights(
    model: Union[str, int] = DEFAULT_ENSEMBLE,
    download_dir: str = ".",
    host: Optional[str] = None,
):
    """Download pre-trained Decima model weights from wandb."""
    if model in ENSEMBLE_MODELS:
        return [download_decima_weights(m, download_dir, host) for m in MODEL_METADATA[model]]

    name = MODEL_METADATA[model]["name"]
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    art = _get_artifact(name, host)
    art.download(str(download_dir))
    return download_dir / f"{name}.safetensors"


def download_decima_metadata(
    metadata: str = DEFAULT_ENSEMBLE,
    download_dir: str = ".",
    host: Optional[str] = None,
):
    """Download pre-trained Decima metadata from wandb."""
    metadata = metadata or DEFAULT_ENSEMBLE
    if metadata in ENSEMBLE_MODELS:
        metadata = MODEL_METADATA[metadata][0]

    metadata_name = MODEL_METADATA[metadata]["metadata"]
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    art = _get_artifact(metadata_name, host)
    art.download(str(download_dir))
    return download_dir / f"{metadata_name}.h5ad"


def cache_decima_data(host: Optional[str] = None):
    """Download all required Decima assets from wandb."""
    login_wandb(host)
    for rep in MODEL_METADATA[DEFAULT_ENSEMBLE]:
        load_decima_model(rep, host=host)
    load_decima_metadata(host=host)
