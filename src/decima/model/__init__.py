from typing import Optional, Union

from decima.hub import load_decima_model
from decima.model.lightning import LightningModel


def load_model(model: Optional[Union[str, int]] = 0, device: str = "cpu"):
    """Load the trained model from a checkpoint path.

    Args:
        model: Path to model checkpoint or replicate number (0-3) for pre-trained models
        device: Device to load model on

    Returns:
        LightningModel: Loaded model

    Examples:
        >>> result = DecimaResult.load()
        >>> result.load_model()  # Load default model (rep0)
        >>> result.load_model(
        ...     model="path/to/checkpoint.ckpt"
        ... )
        >>> result.load_model(
        ...     model=2
        ... )
    """
    if model in {0, 1, 2, 3}:
        _model = load_decima_model(rep=model, device=device)
    else:
        _model = LightningModel.load_from_checkpoint(model, map_location=device)
    return _model
