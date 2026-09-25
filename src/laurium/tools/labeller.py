import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from laurium.tools import labeller_utils

    return labeller_utils, mo


@app.cell
def _(mo):
    mo.md("""
    # Case-by-case labelling tool

    Review one article at a time and record every person mentioned in
    it, along with their name, age and relationship to the article's
    main subject.

    Select an article in the table below to see its full text and edit
    its annotations. Leave a CSV path blank to use the demo dataset.
    """)
    return


@app.cell
def _(mo):
    articles_path = mo.ui.text(
        label="Articles CSV path (leave blank for demo data)",
        placeholder="path/to/articles.csv",
        full_width=True,
    )
    annotations_path = mo.ui.text(
        label="Annotations CSV path (leave blank for demo data)",
        placeholder="path/to/annotations.csv",
        full_width=True,
    )
    article_id_column = mo.ui.text(
        label="Article ID column", value="article_id"
    )
    article_text_column = mo.ui.text(
        label="Article text column", value="article_text"
    )
    name_column = mo.ui.text(label="Name column", value="name")
    age_column = mo.ui.text(label="Age column", value="age")
    relationship_column = mo.ui.text(
        label="Relationship column", value="relationship"
    )
    output_path = mo.ui.text(
        label="Output CSV path",
        value="annotations_output.csv",
        full_width=True,
    )
    mo.vstack(
        [
            articles_path,
            annotations_path,
            article_id_column,
            article_text_column,
            name_column,
            age_column,
            relationship_column,
            output_path,
        ]
    )
    return (
        age_column,
        annotations_path,
        article_id_column,
        article_text_column,
        articles_path,
        name_column,
        output_path,
        relationship_column,
    )


@app.cell
def _(
    article_id_column,
    article_text_column,
    articles_path,
    labeller_utils,
    mo,
):
    mo.stop(
        not article_id_column.value.strip()
        or not article_text_column.value.strip(),
        mo.md(
            "Please provide both an article ID column and an article "
            "text column."
        ),
    )

    try:
        articles = labeller_utils.load_articles(
            path=articles_path.value.strip() or None,
            id_column=article_id_column.value.strip(),
            text_column=article_text_column.value.strip(),
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        mo.stop(True, mo.md(f"**Could not load articles:** {error}"))
    return (articles,)


@app.cell
def _(
    age_column,
    annotations_path,
    article_id_column,
    labeller_utils,
    mo,
    name_column,
    relationship_column,
):
    try:
        initial_annotations = labeller_utils.load_annotations(
            path=annotations_path.value.strip() or None,
            id_column=article_id_column.value.strip(),
            name_column=name_column.value.strip(),
            age_column=age_column.value.strip(),
            relationship_column=relationship_column.value.strip(),
        )
    except (ValueError, FileNotFoundError, OSError) as error:
        mo.stop(True, mo.md(f"**Could not load annotations:** {error}"))
    return (initial_annotations,)


@app.cell
def _(initial_annotations, mo):
    get_annotations, set_annotations = mo.state(initial_annotations)
    return get_annotations, set_annotations


@app.cell
def _(article_id_column, article_text_column, articles, mo):
    article_table = mo.ui.table(
        articles[[article_id_column.value, article_text_column.value]],
        selection="single",
        show_column_summaries=False,
    )
    article_table
    return (article_table,)


@app.cell
def _(article_table, article_text_column, mo):
    mo.stop(
        article_table.value.empty,
        mo.md("*Select an article above to view and edit its annotations.*"),
    )
    selected_article = article_table.value.iloc[0]
    mo.md(f"**Article:**\n\n{selected_article[article_text_column.value]}")
    return (selected_article,)


@app.cell
def _(
    age_column,
    article_id_column,
    get_annotations,
    labeller_utils,
    mo,
    name_column,
    relationship_column,
    selected_article,
):
    current_annotations = get_annotations()
    article_annotations = labeller_utils.filter_annotations_by_article(
        current_annotations,
        selected_article[article_id_column.value],
        article_id_column.value,
    )
    annotation_editor = mo.ui.data_editor(
        article_annotations[
            [
                name_column.value.strip(),
                age_column.value.strip(),
                relationship_column.value.strip(),
            ]
        ]
    ).form(bordered=False)
    annotation_editor
    return annotation_editor, current_annotations


@app.cell
def _(
    annotation_editor,
    article_id_column,
    current_annotations,
    labeller_utils,
    mo,
    selected_article,
    set_annotations,
):
    mo.stop(annotation_editor.value is None)

    updated_annotations = labeller_utils.replace_article_annotations(
        current_annotations,
        selected_article[article_id_column.value],
        annotation_editor.value,
        id_column=article_id_column.value,
    )
    set_annotations(updated_annotations)
    return


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save all annotations to CSV")
    save_button
    return (save_button,)


@app.cell
def _(get_annotations, labeller_utils, mo, output_path, save_button):
    mo.stop(not save_button.value)
    labeller_utils.save_annotations(get_annotations(), output_path.value)
    mo.md(f"Saved annotations to `{output_path.value}`.")
    return


@app.cell
def _(article_id_column, articles, get_annotations, mo):
    current_annotations_summary = get_annotations()
    annotated_articles = current_annotations_summary[
        article_id_column.value
    ].nunique()
    mo.md(
        f"**Progress:** {annotated_articles} / {len(articles)} articles "
        "have at least one annotated person  \n"
        f"**Total people annotated:** {len(current_annotations_summary)}"
    )
    return


if __name__ == "__main__":
    app.run()
