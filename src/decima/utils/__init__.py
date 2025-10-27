import logging
from typing import Optional, List

import torch


def _get_on_off_tasks(result: "DecimaResult", tasks: Optional[List[str]] = None, off_tasks: Optional[List[str]] = None):
    if tasks is None:
        tasks = result.cell_metadata.index.tolist()
    elif isinstance(tasks, str):
        tasks = result.query_cells(tasks)
    if isinstance(off_tasks, str):
        off_tasks = result.query_cells(off_tasks)

    return tasks, off_tasks


def _get_genes(
    result: "DecimaResult",
    genes: Optional[List[str]] = None,
    top_n_markers: Optional[int] = None,
    tasks: Optional[List[str]] = None,
    off_tasks: Optional[List[str]] = None,
):
    if (top_n_markers is not None) and (genes is None):
        all_genes = (
            result.marker_zscores(tasks=tasks, off_tasks=off_tasks)
            .query('task == "on"')
            .sort_values("score", ascending=False)
            .drop_duplicates(subset="gene", keep="first")
            .iloc[:top_n_markers]
            .gene.tolist()
        )
    elif genes is not None:
        if top_n_markers is not None:
            raise ValueError(
                "Cannot specify arguments `genes` and `top_n_markers` at the same time. Only one can be specified."
            )
        all_genes = genes if isinstance(genes, list) else [genes]
    else:
        logger = logging.getLogger("decima")
        all_genes = list(result.genes)
        logger.info(f"No genes provided, using all genes {len(all_genes)} in the result.")

    return all_genes


def get_compute_device(device: Optional[str] = None) -> torch.device:
    """Get the best available device for computation.

    Args:
        device: Optional device specification. If None, automatically selects best available device.

    Returns:
        torch.device: The selected device for computation
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    elif isinstance(device, str) and device.isdigit():
        device = int(device)

    return torch.device(device)
