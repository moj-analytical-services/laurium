"""Helper functions for the label evaluation tool."""

import pandas as pd

ID_COLUMN = "record_id"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
EXPLANATION_COLUMN = "explanation"
ACTUAL_COLUMN = "actual_label"
PREDICTED_COLUMN = "predicted_label"


def create_demo_human_labels() -> pd.DataFrame:
    """Build a small demo dataset of human-assigned labels.

    Returns
    -------
    pd.DataFrame
        A dataframe with `record_id`, `text` and `label` columns,
        representing feedback form submissions triaged to the department
        they relate to.
    """
    return pd.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4, 5, 6],
            TEXT_COLUMN: [
                "The sales team promised a discount that was never "
                "applied to my order.",
                "I still haven't received my P45 despite leaving the "
                "company three weeks ago.",
                "My laptop keeps disconnecting from the VPN throughout "
                "the day.",
                "Reimbursement for last month's travel expenses is "
                "still pending.",
                "The lift to the fourth floor has been out of service "
                "for over a week.",
                "Our account manager never followed up after the demo call.",
            ],
            LABEL_COLUMN: [
                "Sales",
                "HR",
                "IT",
                "Finance",
                "Facilities",
                "Sales",
            ],
        }
    )


def create_demo_model_labels() -> pd.DataFrame:
    """Build a small demo dataset of model-assigned labels.

    Returns
    -------
    pd.DataFrame
        A dataframe with `record_id`, `label` and `explanation` columns,
        matching the output of `laurium.decoder_models.create_label_model`.
    """
    return pd.DataFrame(
        {
            ID_COLUMN: [1, 2, 3, 4, 5, 6],
            LABEL_COLUMN: [
                "Sales",
                "Finance",
                "IT",
                "Finance",
                "IT",
                "Sales",
            ],
            EXPLANATION_COLUMN: [
                "Mentions a discount and an order, consistent with a "
                "sales enquiry.",
                "Mentions a P45, a payroll document handled by Finance.",
                "References a VPN connection issue, a clear IT support "
                "request.",
                "Describes an outstanding expense reimbursement.",
                "Mentions equipment being 'out of service', which the "
                "model associated with an IT fault.",
                "Refers to an account manager and a demo call, "
                "indicating a sales relationship.",
            ],
        }
    )


def load_labels(
    path: str | None,
    demo: pd.DataFrame,
    label_column: str,
) -> pd.DataFrame:
    """Load a labelled dataset from CSV, or fall back to demo data.

    Parameters
    ----------
    path : str, optional
        A local path or URI readable by `pandas.read_csv`. If `None` or
        empty, `demo` is used instead.
    demo : pd.DataFrame
        Demo data to use when `path` is not provided.
    label_column : str
        Name of the column expected to contain the label.

    Returns
    -------
    pd.DataFrame
        The loaded (or demo) dataframe.

    Raises
    ------
    ValueError
        If `label_column` is not present in the loaded data.
    """
    data = pd.read_csv(path) if path else demo.copy()

    if label_column not in data.columns:
        raise ValueError(
            f"Column '{label_column}' not found; available columns are "
            f"{list(data.columns)}."
        )

    return data


def merge_labels(
    human_data: pd.DataFrame,
    model_data: pd.DataFrame,
    id_column: str,
    human_label_column: str,
    model_label_column: str,
) -> pd.DataFrame:
    """Join human and model labels on a shared identifier.

    Parameters
    ----------
    human_data : pd.DataFrame
        Human-labelled data.
    model_data : pd.DataFrame
        Model-labelled data.
    id_column : str
        Name of the column shared by both datasets that uniquely
        identifies each record.
    human_label_column : str
        Name of the label column in `human_data`.
    model_label_column : str
        Name of the label column in `model_data`.

    Returns
    -------
    pd.DataFrame
        An inner join of `human_data` and `model_data` on `id_column`,
        with the label columns renamed to `actual_label` and
        `predicted_label`, all other columns preserved (suffixed with
        `_human`/`_model` where both sides share a column name), and
        rows missing either label removed.

    Raises
    ------
    ValueError
        If `id_column` or the relevant label column is missing from
        either dataset.
    """
    for name, data, column in (
        ("human", human_data, human_label_column),
        ("model", model_data, model_label_column),
    ):
        missing = [c for c in (id_column, column) if c not in data.columns]
        if missing:
            raise ValueError(
                f"{name.capitalize()} data is missing required column(s): "
                f"{missing}"
            )

    human = human_data.rename(columns={human_label_column: ACTUAL_COLUMN})
    model = model_data.rename(columns={model_label_column: PREDICTED_COLUMN})

    merged = pd.merge(
        human, model, on=id_column, how="inner", suffixes=("_human", "_model")
    )
    return merged.dropna(subset=[ACTUAL_COLUMN, PREDICTED_COLUMN])


def summarize_match_counts(
    human_data: pd.DataFrame,
    model_data: pd.DataFrame,
    merged: pd.DataFrame,
) -> dict[str, int]:
    """Summarize dataset sizes and how many records matched across both.

    Parameters
    ----------
    human_data : pd.DataFrame
        Human-labelled data.
    model_data : pd.DataFrame
        Model-labelled data.
    merged : pd.DataFrame
        The result of `merge_labels` for `human_data` and `model_data`.

    Returns
    -------
    dict[str, int]
        Counts for `human_rows`, `model_rows` and `matched_rows`.
    """
    return {
        "human_rows": len(human_data),
        "model_rows": len(model_data),
        "matched_rows": len(merged),
    }


def build_confusion_counts(
    merged: pd.DataFrame,
    actual_column: str = ACTUAL_COLUMN,
    predicted_column: str = PREDICTED_COLUMN,
) -> pd.DataFrame:
    """Compute confusion matrix counts for every actual/predicted combination.

    Parameters
    ----------
    merged : pd.DataFrame
        A dataframe containing `actual_column` and `predicted_column`.
    actual_column : str, default "actual_label"
        Name of the column containing the ground-truth label.
    predicted_column : str, default "predicted_label"
        Name of the column containing the predicted label.

    Returns
    -------
    pd.DataFrame
        One row per (actual, predicted) label combination with a `count`
        column, including zero-count combinations, sorted by label.
    """
    labels = sorted(set(merged[actual_column]) | set(merged[predicted_column]))

    counts = (
        merged.groupby([actual_column, predicted_column])
        .size()
        .rename("count")
        .reset_index()
    )

    full_index = pd.MultiIndex.from_product(
        [labels, labels], names=[actual_column, predicted_column]
    )
    return (
        counts.set_index([actual_column, predicted_column])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )


def filter_examples_by_cell(
    merged: pd.DataFrame,
    actual_label: str | None,
    predicted_label: str | None,
    actual_column: str = ACTUAL_COLUMN,
    predicted_column: str = PREDICTED_COLUMN,
) -> pd.DataFrame:
    """Filter merged examples to a single selected confusion matrix cell.

    Parameters
    ----------
    merged : pd.DataFrame
        The result of `merge_labels`.
    actual_label : str, optional
        The ground-truth label of the selected cell. If `None`, all rows
        are returned.
    predicted_label : str, optional
        The predicted label of the selected cell. If `None`, all rows are
        returned.
    actual_column : str, default "actual_label"
        Name of the column containing the ground-truth label.
    predicted_column : str, default "predicted_label"
        Name of the column containing the predicted label.

    Returns
    -------
    pd.DataFrame
        All of `merged` if no cell is selected, otherwise only the rows
        matching both `actual_label` and `predicted_label`.
    """
    if actual_label is None or predicted_label is None:
        return merged

    return merged[
        (merged[actual_column] == actual_label)
        & (merged[predicted_column] == predicted_label)
    ]


def select_display_columns(
    merged: pd.DataFrame,
    id_column: str,
    text_column: str | None = None,
    explanation_column: str | None = None,
    actual_column: str = ACTUAL_COLUMN,
    predicted_column: str = PREDICTED_COLUMN,
) -> pd.DataFrame:
    """Select and order columns for displaying evaluation examples.

    Parameters
    ----------
    merged : pd.DataFrame
        The result of `merge_labels` (optionally filtered by
        `filter_examples_by_cell`).
    id_column : str
        Name of the join key column.
    text_column : str, optional
        Name of a free-text/example column to include, if present.
    explanation_column : str, optional
        Name of a model explanation column to include, if present.
    actual_column : str, default "actual_label"
        Name of the column containing the ground-truth label.
    predicted_column : str, default "predicted_label"
        Name of the column containing the predicted label.

    Returns
    -------
    pd.DataFrame
        `merged` restricted to the id column, any available text and
        explanation columns, and the actual/predicted label columns, in
        a fixed display order.
    """
    columns = [id_column]
    for column in (text_column, explanation_column):
        if column and column in merged.columns:
            columns.append(column)
    columns += [actual_column, predicted_column]
    return merged[columns]
