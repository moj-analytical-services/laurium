"""Loading data from S3 and locally."""

import itertools as it
from copy import deepcopy
from pathlib import Path
from typing import Any, Generator, Iterable, Optional

import pandas as pd
import pydbtools


def data_loader(
    chunk_size: int,
    query_template_path: str | Path,
    max_chunks: Optional[int] = None,
    **template_args: Any,
) -> Generator[pd.DataFrame, None, None]:
    """Generate chunks of free text data.

    Parameters
    ----------
    chunk_size : int
        The number of entries to include in each chunk
    query_template_path: str or pathlib.Path
        The path to the SQL query template to fetch. The SQL query must define
        `offset` and `chunk_size` variables.
    max_chunks : int, optional
        The number of chunks to yield. If None, all data will be loaded
        Default is None
    **template_args
        Any additional arguments to be passed to the Jinja templating engine.

    Yields
    ------
    Generator[pd.DataFrame]
        Data frames with free text entries, chunked into chunk_size chunks.
    """
    # Read query template
    with open(query_template_path, encoding="utf-8") as f:
        template = f.read()

    # Loop over records in steps of chunk_size
    if max_chunks is not None:
        # Loop until final chunk
        offsets: Iterable[int] = range(0, chunk_size * max_chunks, chunk_size)
    else:
        # Loop infinitely in steps of chunk_size
        offsets = it.count(step=chunk_size)

    jinja_args = deepcopy(template_args)
    jinja_args["chunk_size"] = chunk_size

    for offset in offsets:
        # Populate query
        jinja_args["offset"] = offset
        sql = pydbtools.render_sql_template(
            template,
            jinja_args=jinja_args,
        )

        # Load results
        # NB ctas_approach=False required because of issues with datetime cols
        # See aws/aws-sdk-pandas#2252
        results = pydbtools.read_sql_query(sql, ctas_approach=False)

        # No results - have passed the final chunk
        if len(results.index) == 0:
            break

        yield results
