import logging
from typing import List, Optional

from decima.core.result import DecimaResult


def _get_on_off_tasks(result: DecimaResult, tasks: Optional[List[str]] = None, off_tasks: Optional[List[str]] = None):
    if tasks is None:
        tasks = result.cell_metadata.index.tolist()
    elif isinstance(tasks, str):
        tasks = result.query_cells(tasks)
    if isinstance(off_tasks, str):
        off_tasks = result.query_cells(off_tasks)

    return tasks, off_tasks


def _get_genes(
    result: DecimaResult,
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
        all_genes = genes
    else:
        logger = logging.getLogger("decima")
        all_genes = list(result.genes)
        logger.info(f"No genes provided, using all genes {len(all_genes)} in the result.")

    return all_genes
