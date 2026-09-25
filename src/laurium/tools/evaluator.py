import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    from laurium.tools import evaluator_utils

    return alt, evaluator_utils, mo


@app.cell
def _(mo):
    mo.md("""
    # Label evaluation tool

    Compare human-labelled and model-labelled data to see how well a
    model's predictions match human judgement, either from your own
    CSV files or from a small built-in demo dataset.

    Leave a CSV path blank to use demo data for that input.
    """)
    return


@app.cell
def _(mo):
    human_path = mo.ui.text(
        label="Human-labelled CSV path (leave blank for demo data)",
        placeholder="path/to/human_labels.csv",
        full_width=True,
    )
    model_path = mo.ui.text(
        label="Model-labelled CSV path (leave blank for demo data)",
        placeholder="path/to/model_labels.csv",
        full_width=True,
    )
    id_column = mo.ui.text(label="Join key column", value="record_id")
    human_label_column = mo.ui.text(label="Human label column", value="label")
    model_label_column = mo.ui.text(label="Model label column", value="label")
    text_column = mo.ui.text(label="Free text column (optional)", value="text")
    explanation_column = mo.ui.text(
        label="Explanation column (optional)", value="explanation"
    )
    mo.vstack(
        [
            human_path,
            model_path,
            id_column,
            human_label_column,
            model_label_column,
            text_column,
            explanation_column,
        ]
    )
    return (
        explanation_column,
        human_label_column,
        human_path,
        id_column,
        model_label_column,
        model_path,
        text_column,
    )


@app.cell
def _(
    evaluator_utils,
    human_label_column,
    human_path,
    mo,
    model_label_column,
    model_path,
):
    try:
        human_data = evaluator_utils.load_labels(
            human_path.value.strip() or None,
            evaluator_utils.create_demo_human_labels(),
            human_label_column.value.strip(),
        )
        model_data = evaluator_utils.load_labels(
            model_path.value.strip() or None,
            evaluator_utils.create_demo_model_labels(),
            model_label_column.value.strip(),
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        mo.stop(True, mo.md(f"**Could not load data:** {error}"))
    return human_data, model_data


@app.cell
def _(
    evaluator_utils,
    human_data,
    human_label_column,
    id_column,
    mo,
    model_data,
    model_label_column,
):
    try:
        merged = evaluator_utils.merge_labels(
            human_data,
            model_data,
            id_column.value.strip(),
            human_label_column.value.strip(),
            model_label_column.value.strip(),
        )
    except ValueError as error:
        mo.stop(True, mo.md(f"**Could not join data:** {error}"))

    mo.stop(
        merged.empty,
        mo.md(
            "No matching, labelled records were found between the two files."
        ),
    )
    return (merged,)


@app.cell
def _(evaluator_utils, human_data, merged, mo, model_data):
    counts = evaluator_utils.summarize_match_counts(
        human_data, model_data, merged
    )
    mo.md(
        f"**Human rows:** {counts['human_rows']}  \n"
        f"**Model rows:** {counts['model_rows']}  \n"
        f"**Matched rows:** {counts['matched_rows']}"
    )
    return


@app.cell
def _(evaluator_utils, merged):
    confusion_counts = evaluator_utils.build_confusion_counts(merged)
    return (confusion_counts,)


@app.cell
def _(alt, confusion_counts, mo):
    chart = mo.ui.altair_chart(
        alt.Chart(confusion_counts)
        .mark_rect()
        .encode(
            alt.X("predicted_label:N").title("Predicted"),
            alt.Y("actual_label:N").title("Actual"),
            alt.Color("count:Q").scale(scheme="greenblue"),
        )
        .properties(height=400, width=400)
        + alt.Chart(confusion_counts)
        .mark_text()
        .encode(
            alt.X("predicted_label:N").title("Predicted"),
            alt.Y("actual_label:N").title("Actual"),
            text="count:Q",
        )
    )
    chart
    return (chart,)


@app.cell
def _(chart, evaluator_utils, mo):
    selected_cell = chart.value
    if selected_cell.empty:
        selected_actual = None
        selected_predicted = None
        mo.output.append(
            mo.md(
                "*Showing all matched examples. Click a cell above to filter.*"
            )
        )
    else:
        selected_actual = selected_cell[evaluator_utils.ACTUAL_COLUMN].iloc[0]
        selected_predicted = selected_cell[
            evaluator_utils.PREDICTED_COLUMN
        ].iloc[0]
        mo.output.append(
            mo.md(
                f"**Showing examples where actual = `{selected_actual}` and "
                f"predicted = `{selected_predicted}`.**"
            )
        )
    return selected_actual, selected_predicted


@app.cell
def _(
    evaluator_utils,
    explanation_column,
    id_column,
    merged,
    mo,
    selected_actual,
    selected_predicted,
    text_column,
):
    filtered_examples = evaluator_utils.filter_examples_by_cell(
        merged, selected_actual, selected_predicted
    )
    display_data = evaluator_utils.select_display_columns(
        filtered_examples,
        id_column.value.strip(),
        text_column.value.strip() or None,
        explanation_column.value.strip() or None,
    )

    wrapped_columns = [
        column
        for column in (
            text_column.value.strip(),
            explanation_column.value.strip(),
        )
        if column and column in display_data.columns
    ]

    mo.ui.table(
        display_data,
        show_column_summaries=False,
        wrapped_columns=wrapped_columns,
    )
    return


if __name__ == "__main__":
    app.run()
