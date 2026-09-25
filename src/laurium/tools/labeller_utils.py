"""Helper functions for the interactive case-by-case labelling tool."""

from typing import Any

import pandas as pd

ARTICLE_ID_COLUMN = "article_id"
ARTICLE_TEXT_COLUMN = "article_text"
ANNOTATION_ID_COLUMN = "annotation_id"
NAME_COLUMN = "name"
AGE_COLUMN = "age"
RELATIONSHIP_COLUMN = "relationship"


def create_demo_articles() -> pd.DataFrame:
    """Build a small demo dataset of newspaper-style articles.

    Returns
    -------
    pd.DataFrame
        A dataframe with `article_id` and `article_text` columns.
    """
    return pd.DataFrame(
        {
            ARTICLE_ID_COLUMN: [1, 2, 3],
            ARTICLE_TEXT_COLUMN: [
                "Son, 4, rescues mother - Billy, aged 4, called 999 to "
                "save the life of his mother Agneta, 37. Her brother "
                "Simon praised the bravery of the boy.",
                "Local bakery marks 50 years in business - Owner Maria, "
                "62, has run the shop since taking over from her father "
                "Joseph. Her daughter Elena now manages the accounts.",
                "Teenager reunited with lost dog after three days - "
                "Isaac, 15, was overjoyed to be reunited with his dog "
                "Max. His grandmother Ruth had helped search the "
                "neighbourhood.",
            ],
        }
    )


def create_demo_annotations() -> pd.DataFrame:
    """Build a small demo dataset of extracted person annotations.

    Returns
    -------
    pd.DataFrame
        A dataframe with `annotation_id`, `article_id`, `name`, `age`
        and `relationship` columns. Only the first demo article (see
        `create_demo_articles`) is pre-annotated, as a worked example.
    """
    return pd.DataFrame(
        {
            ANNOTATION_ID_COLUMN: ["1-0", "1-1", "1-2"],
            ARTICLE_ID_COLUMN: [1, 1, 1],
            NAME_COLUMN: ["Billy", "Agneta", "Simon"],
            AGE_COLUMN: pd.array([4, 37, None], dtype="object"),
            RELATIONSHIP_COLUMN: ["son", "mother", "uncle"],
        }
    )


def load_articles(
    path: str | None,
    id_column: str,
    text_column: str,
) -> pd.DataFrame:
    """Load articles to be annotated from a CSV path, or use demo data.

    Parameters
    ----------
    path : str, optional
        A local path or URI readable by `pandas.read_csv`. If `None` or
        empty, the built-in demo articles are used instead.
    id_column : str
        Name of the column that uniquely identifies each article.
    text_column : str
        Name of the column containing each article's full text.

    Returns
    -------
    pd.DataFrame
        The loaded (or demo) articles dataframe.

    Raises
    ------
    ValueError
        If `id_column` or `text_column` are missing, or `id_column` does
        not contain unique values.
    """
    data = pd.read_csv(path) if path else create_demo_articles()

    missing = [
        column
        for column in (id_column, text_column)
        if column not in data.columns
    ]
    if missing:
        raise ValueError(
            f"Article data is missing required column(s): {missing}"
        )

    if data[id_column].duplicated().any():
        raise ValueError(f"Column '{id_column}' must contain unique values.")

    return data


def load_annotations(
    path: str | None,
    id_column: str,
    name_column: str,
    age_column: str,
    relationship_column: str,
    annotation_id_column: str = ANNOTATION_ID_COLUMN,
) -> pd.DataFrame:
    """Load existing annotations from a CSV path, or use demo annotations.

    Parameters
    ----------
    path : str, optional
        A local path or URI readable by `pandas.read_csv`. If `None` or
        empty, the built-in demo annotations are used instead.
    id_column : str
        Name of the column linking each annotation to an article.
    name_column : str
        Name of the column holding each person's name.
    age_column : str
        Name of the column holding each person's age (may be blank).
    relationship_column : str
        Name of the column holding each person's relationship to the
        article's main subject.
    annotation_id_column : str, default "annotation_id"
        Name of the column giving each annotation row a unique, stable
        identifier. Generated automatically if not already present.

    Returns
    -------
    pd.DataFrame
        The loaded (or demo) annotations dataframe, with
        `annotation_id_column` added if it was missing.

    Raises
    ------
    ValueError
        If any required column is missing, or `annotation_id_column`
        already exists but contains duplicate values.
    """
    data = pd.read_csv(path) if path else create_demo_annotations()

    required = (id_column, name_column, age_column, relationship_column)
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(
            f"Annotation data is missing required column(s): {missing}"
        )

    if annotation_id_column not in data.columns:
        data = data.copy()
        data[annotation_id_column] = _generate_annotation_ids(data, id_column)
    elif data[annotation_id_column].duplicated().any():
        raise ValueError(
            f"Column '{annotation_id_column}' must contain unique values."
        )

    return data


def _generate_annotation_ids(data: pd.DataFrame, id_column: str) -> list[str]:
    """Generate stable per-article annotation identifiers.

    Parameters
    ----------
    data : pd.DataFrame
        Annotation rows, in the order they should be numbered.
    id_column : str
        Name of the column identifying the article each row belongs to.

    Returns
    -------
    list[str]
        One `f"{article_id}-{position}"` identifier per row, where
        `position` restarts at 0 for each distinct article.
    """
    counters: dict[Any, int] = {}
    ids = []
    for article_id in data[id_column]:
        position = counters.get(article_id, 0)
        ids.append(f"{article_id}-{position}")
        counters[article_id] = position + 1
    return ids


def filter_annotations_by_article(
    annotations: pd.DataFrame,
    article_id: Any,
    id_column: str,
) -> pd.DataFrame:
    """Select the annotation rows belonging to a single article.

    Parameters
    ----------
    annotations : pd.DataFrame
        The full annotations dataset.
    article_id : Any
        The identifier of the article to filter to.
    id_column : str
        Name of the column identifying the article in `annotations`.

    Returns
    -------
    pd.DataFrame
        Only the rows of `annotations` matching `article_id`, with a
        fresh row index.
    """
    return annotations[annotations[id_column] == article_id].reset_index(
        drop=True
    )


def replace_article_annotations(
    annotations: pd.DataFrame,
    article_id: Any,
    edited_rows: pd.DataFrame,
    id_column: str,
    annotation_id_column: str = ANNOTATION_ID_COLUMN,
) -> pd.DataFrame:
    """Replace all annotation rows for one article with an edited set.

    Parameters
    ----------
    annotations : pd.DataFrame
        The full annotations dataset.
    article_id : Any
        The identifier of the article whose annotation rows are being
        replaced.
    edited_rows : pd.DataFrame
        The new set of annotation rows for `article_id` (for example,
        one row per person mentioned in the article). May be empty.
    id_column : str
        Name of the column identifying the article in both
        `annotations` and via `article_id`.
    annotation_id_column : str, default "annotation_id"
        Name of the column used to give each annotation row a unique,
        stable identifier.

    Returns
    -------
    pd.DataFrame
        `annotations` with all rows for `article_id` removed and
        replaced by `edited_rows`, each assigned a fresh
        `annotation_id_column` value and `id_column` set to
        `article_id`.
    """
    remaining = annotations[annotations[id_column] != article_id]

    new_rows = edited_rows.copy().reset_index(drop=True)
    new_rows[id_column] = article_id
    new_rows[annotation_id_column] = [
        f"{article_id}-{position}" for position in range(len(new_rows))
    ]

    updated = pd.concat([remaining, new_rows], ignore_index=True)
    return updated[list(annotations.columns)]


def save_annotations(data: pd.DataFrame, path: str) -> None:
    """Save the current annotations dataset to a CSV file.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to save.
    path : str
        Destination path for the CSV file.
    """
    data.to_csv(path, index=False)
