"""
Query Cell CLI.

This module contains the CLI for querying the cell metadata.

`decima query-cell` is the command for querying the cell metadata.

Examples:

    >>> decima query-cell 'cell_type == "classical monocyte"'
    ...

    >>> decima query-cell 'cell_type == "classical monocyte" and disease == "healthy" and tissue == "blood"'
    ...

    >>> decima query-cell 'cell_type.str.contains("monocyte") and disease == "healthy"'
    ...
"""

import click
from decima.constants import DEFAULT_ENSEMBLE
from decima.cli.callback import parse_metadata
from decima.core.result import DecimaResult


@click.command()
@click.argument("query", default="")
@click.option(
    "--metadata",
    default=None,
    callback=parse_metadata,
    help=f"Path to the metadata anndata file or name of the model. Default: {DEFAULT_ENSEMBLE}.",
)
def cli_query_cell(query="", metadata=None):
    """
    Query a cell using query string

    Examples:

        >>> decima query-cell 'cell_type == "classical monocyte"'
        ...

        >>> decima query-cell 'cell_type == "classical monocyte" and disease == "healthy" and tissue == "blood"'
        ...

        >>> decima query-cell 'cell_type.str.contains("monocyte") and disease == "healthy"'
        ...

    """
    result = DecimaResult.load(metadata)
    df = result.cell_metadata

    if query != "":
        df = df.query(query)

    print(df.to_csv(sep="\t", index=True, header=True))
