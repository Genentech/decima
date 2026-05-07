import logging
from pathlib import Path
from typing import Union
import genomepy
from huggingface_hub import hf_hub_download
from decima.constants import (
    DEFAULT_ENSEMBLE,
    ENSEMBLE_MODELS,
    MODEL_METADATA,
    HF_MODEL_REPO,
    HF_DATA_REPO,
    HF_METADATA_FILENAME,
)
from decima.hub import load_decima_model, load_decima_metadata

logger = logging.getLogger("decima")


def cache_hg38():
    """Download hg38 genome from UCSC."""
    logger.info("Downloading hg38 genome...")
    genomepy.install_genome(provider="url", name="http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz")


def cache_decima_weights():
    """Download pre-trained Decima model weights from HuggingFace."""
    logger.info("Downloading Decima model weights...")
    for rep in MODEL_METADATA[DEFAULT_ENSEMBLE]:
        load_decima_model(rep)


def cache_decima_metadata():
    """Download pre-trained Decima metadata from HuggingFace."""
    logger.info("Downloading Decima metadata...")
    load_decima_metadata()


def cache_decima_data():
    """Download all required data for Decima."""
    cache_hg38()
    cache_decima_weights()
    cache_decima_metadata()


def download_decima_weights(model: Union[str, int] = DEFAULT_ENSEMBLE, download_dir: str = "."):
    """Download pre-trained Decima model weights from HuggingFace to a local directory.

    Args:
        model: Model name or replicate number.
        download_dir: Directory to save the model weights.

    Returns:
        Path to the downloaded model weights.
    """
    if model in ENSEMBLE_MODELS:
        return [download_decima_weights(m, download_dir) for m in MODEL_METADATA[model]]

    name = MODEL_METADATA[model]["name"]
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Decima model weights for {model} to {download_dir / f'{name}.safetensors'}")
    return hf_hub_download(repo_id=HF_MODEL_REPO, filename=f"{name}.safetensors", local_dir=str(download_dir))


def download_decima_metadata(metadata: str = DEFAULT_ENSEMBLE, download_dir: str = "."):
    """Download pre-trained Decima metadata from HuggingFace to a local directory.

    Args:
        metadata: Model name to select metadata for.
        download_dir: Directory to save the metadata.

    Returns:
        Path to the downloaded metadata.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Decima metadata to {download_dir / HF_METADATA_FILENAME}")
    return hf_hub_download(
        repo_id=HF_DATA_REPO, filename=HF_METADATA_FILENAME, repo_type="dataset", local_dir=str(download_dir)
    )


def download_decima(model: str = DEFAULT_ENSEMBLE, download_dir: str = "."):
    """Download all required data for Decima.

    Args:
        model: Model name or replicate number.
        download_dir: Directory to save the model weights and metadata.

    Returns:
        Path to the download directory.
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading Decima model weights and metadata to {download_dir}:")
    download_decima_weights(model, download_dir)
    download_decima_metadata(model, download_dir)
    return download_dir
