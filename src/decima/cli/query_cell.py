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
from decima.core.result import DecimaResult


@click.command()
@click.argument("query", default="")
def cli_query_cell(query=""):
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
    result = DecimaResult.load()
    df = result.cell_metadata

    if query != "":
        df = df.query(query)

    print(df.to_csv(sep="\t", index=True, header=True))
