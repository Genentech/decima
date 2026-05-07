from pathlib import Path
from typing import Union, Optional, List
import warnings
import anndata
from huggingface_hub import hf_hub_download
from decima.constants import (
    DEFAULT_ENSEMBLE,
    ENSEMBLE_MODELS,
    MODEL_METADATA,
    HF_MODEL_REPO,
    HF_DATA_REPO,
    HF_METADATA_FILENAME,
)
from decima.model.lightning import LightningModel, EnsembleLightningModel


def load_decima_model(model: Union[str, int, List[str]] = DEFAULT_ENSEMBLE, device: Optional[str] = None):
    """Load a pre-trained Decima model from HuggingFace or a local path.

    Args:
        model: Model identifier or path. Can be:
            - int: Replicate number (0-3)
            - str: Model name from MODEL_METADATA
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

    if model in ENSEMBLE_MODELS:
        return EnsembleLightningModel(
            [load_decima_model(model_name, device) for model_name in MODEL_METADATA[model]],
            name=model,
        )

    if isinstance(model, list):
        if len(model) == 1:
            return load_decima_model(model[0], device)
        return EnsembleLightningModel([load_decima_model(path, device) for path in model], name=model)

    if isinstance(model, str) and Path(model).exists():
        if model.endswith(".ckpt"):
            return LightningModel.load_from_checkpoint(model, map_location=device)
        return LightningModel.load_safetensor(model, device=device)

    if model in MODEL_METADATA:
        if "model_path" in MODEL_METADATA[model]:
            return load_decima_model(MODEL_METADATA[model]["model_path"], device)
        name = MODEL_METADATA[model]["name"]
        cached = hf_hub_download(repo_id=HF_MODEL_REPO, filename=f"{name}.safetensors")
        return LightningModel.load_safetensor(cached, device=device)

    raise ValueError(
        f"Invalid model: {model}. Must be a known model name {list(MODEL_METADATA.keys())}, "
        "a local path, or a list of local paths."
    )


def load_decima_metadata(name_or_path: Optional[str] = None):
    """Load the Decima metadata from HuggingFace or a local path.

    Args:
        name_or_path: Path to local metadata file or model name. If None, downloads from HuggingFace.

    Returns:
        An AnnData object containing the Decima metadata.
    """
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
                f"Metadata path `{metadata['metadata_path']}` not found. Downloading from HuggingFace."
            )

    cached = hf_hub_download(repo_id=HF_DATA_REPO, filename=HF_METADATA_FILENAME, repo_type="dataset")
    return anndata.read_h5ad(cached)
