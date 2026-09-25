"""Test helper functions for the interactive labelling tool."""

import pandas as pd
import pytest

from laurium.tools import labeller_utils


def test_create_demo_articles_has_expected_columns():
    """Test that demo articles have the expected columns and unique ids."""
    data = labeller_utils.create_demo_articles()

    assert list(data.columns) == ["article_id", "article_text"]
    assert data["article_id"].is_unique


def test_create_demo_annotations_only_covers_first_article():
    """Test that demo annotations pre-fill the first article as an example."""
    data = labeller_utils.create_demo_annotations()

    assert list(data.columns) == [
        "annotation_id",
        "article_id",
        "name",
        "age",
        "relationship",
    ]
    assert set(data["article_id"]) == {1}
    assert list(data["name"]) == ["Billy", "Agneta", "Simon"]
    assert data.loc[data["name"] == "Simon", "age"].iloc[0] is None


def test_load_articles_returns_demo_data_when_no_path_given():
    """Test that a blank path falls back to demo articles."""
    data = labeller_utils.load_articles(
        path=None, id_column="article_id", text_column="article_text"
    )

    pd.testing.assert_frame_equal(data, labeller_utils.create_demo_articles())


def test_load_articles_reads_csv(tmp_path):
    """Test that a CSV path is read directly."""
    csv_path = tmp_path / "articles.csv"
    pd.DataFrame({"id": [1, 2], "body": ["first", "second"]}).to_csv(
        csv_path, index=False
    )

    data = labeller_utils.load_articles(
        path=str(csv_path), id_column="id", text_column="body"
    )

    assert list(data["body"]) == ["first", "second"]


def test_load_articles_missing_columns_raises(tmp_path):
    """Test that missing required columns raise a ValueError."""
    csv_path = tmp_path / "articles.csv"
    pd.DataFrame({"id": [1], "body": ["first"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="missing required column"):
        labeller_utils.load_articles(
            path=str(csv_path), id_column="id", text_column="missing_column"
        )


def test_load_articles_duplicate_ids_raises(tmp_path):
    """Test that duplicate article ids raise a ValueError."""
    csv_path = tmp_path / "articles.csv"
    pd.DataFrame({"id": [1, 1], "body": ["first", "second"]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="unique values"):
        labeller_utils.load_articles(
            path=str(csv_path), id_column="id", text_column="body"
        )


def test_load_annotations_returns_demo_data_when_no_path_given():
    """Test that a blank path falls back to demo annotations."""
    data = labeller_utils.load_annotations(
        path=None,
        id_column="article_id",
        name_column="name",
        age_column="age",
        relationship_column="relationship",
    )

    pd.testing.assert_frame_equal(
        data, labeller_utils.create_demo_annotations()
    )


def test_load_annotations_generates_ids_when_missing(tmp_path):
    """Test that annotation ids are generated per-article when absent."""
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "article_id": [1, 1, 2],
            "name": ["Billy", "Agneta", "Maria"],
            "age": [4, 37, 62],
            "relationship": ["son", "mother", "owner"],
        }
    ).to_csv(csv_path, index=False)

    data = labeller_utils.load_annotations(
        path=str(csv_path),
        id_column="article_id",
        name_column="name",
        age_column="age",
        relationship_column="relationship",
    )

    assert list(data["annotation_id"]) == ["1-0", "1-1", "2-0"]


def test_load_annotations_missing_columns_raises(tmp_path):
    """Test that missing required columns raise a ValueError."""
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame({"article_id": [1], "name": ["Billy"]}).to_csv(
        csv_path, index=False
    )

    with pytest.raises(ValueError, match="missing required column"):
        labeller_utils.load_annotations(
            path=str(csv_path),
            id_column="article_id",
            name_column="name",
            age_column="age",
            relationship_column="relationship",
        )


def test_load_annotations_duplicate_annotation_id_raises(tmp_path):
    """Test that duplicate annotation ids raise a ValueError."""
    csv_path = tmp_path / "annotations.csv"
    pd.DataFrame(
        {
            "annotation_id": ["a", "a"],
            "article_id": [1, 1],
            "name": ["Billy", "Agneta"],
            "age": [4, 37],
            "relationship": ["son", "mother"],
        }
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="unique values"):
        labeller_utils.load_annotations(
            path=str(csv_path),
            id_column="article_id",
            name_column="name",
            age_column="age",
            relationship_column="relationship",
        )


def test_filter_annotations_by_article_returns_only_matching_rows():
    """Test that filtering selects only the rows for the given article."""
    annotations = labeller_utils.create_demo_annotations()

    result = labeller_utils.filter_annotations_by_article(
        annotations, 1, "article_id"
    )

    assert len(result) == 3
    assert set(result["article_id"]) == {1}


def test_filter_annotations_by_article_empty_for_unannotated_article():
    """Test that an article with no annotations returns an empty frame."""
    annotations = labeller_utils.create_demo_annotations()

    result = labeller_utils.filter_annotations_by_article(
        annotations, 2, "article_id"
    )

    assert result.empty


def test_replace_article_annotations_updates_only_selected_article():
    """Test that replacing one article's rows leaves other articles intact."""
    annotations = labeller_utils.create_demo_annotations()
    edited_rows = pd.DataFrame(
        {
            "name": ["Billy", "Agneta"],
            "age": [4, 37],
            "relationship": ["son", "mother"],
        }
    )

    result = labeller_utils.replace_article_annotations(
        annotations, 1, edited_rows, id_column="article_id"
    )

    assert len(result) == 2
    assert set(result["name"]) == {"Billy", "Agneta"}
    assert list(result["annotation_id"]) == ["1-0", "1-1"]


def test_replace_article_annotations_supports_adding_and_removing_rows():
    """Test that the replaced set can grow, shrink, or start empty."""
    annotations = labeller_utils.create_demo_annotations()

    grown = labeller_utils.replace_article_annotations(
        annotations,
        1,
        pd.DataFrame(
            {
                "name": ["Billy", "Agneta", "Simon", "Extra Person"],
                "age": [4, 37, None, None],
                "relationship": ["son", "mother", "uncle", "neighbour"],
            }
        ),
        id_column="article_id",
    )
    assert len(grown[grown["article_id"] == 1]) == 4

    emptied = labeller_utils.replace_article_annotations(
        annotations,
        1,
        pd.DataFrame(columns=["name", "age", "relationship"]),
        id_column="article_id",
    )
    assert emptied[emptied["article_id"] == 1].empty


def test_replace_article_annotations_preserves_other_articles():
    """Test that editing one article does not affect another article."""
    annotations = pd.concat(
        [
            labeller_utils.create_demo_annotations(),
            pd.DataFrame(
                {
                    "annotation_id": ["2-0"],
                    "article_id": [2],
                    "name": ["Maria"],
                    "age": [62],
                    "relationship": ["owner"],
                }
            ),
        ],
        ignore_index=True,
    )

    result = labeller_utils.replace_article_annotations(
        annotations,
        1,
        pd.DataFrame({"name": ["Billy"], "age": [4], "relationship": ["son"]}),
        id_column="article_id",
    )

    assert list(result[result["article_id"] == 2]["name"]) == ["Maria"]


def test_save_annotations_writes_csv(tmp_path):
    """Test that the dataset is saved to CSV without the pandas index."""
    output_path = tmp_path / "output.csv"
    data = labeller_utils.create_demo_annotations()

    labeller_utils.save_annotations(data, str(output_path))

    saved = pd.read_csv(output_path)
    assert list(saved["name"]) == list(data["name"])
