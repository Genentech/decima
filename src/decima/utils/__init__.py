from typing import Optional
import torch


def get_compute_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.

    Args:
        device: Optional device specification. If None, automatically selects best available device.

    Returns:
        torch.device: The selected device for computation
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)
