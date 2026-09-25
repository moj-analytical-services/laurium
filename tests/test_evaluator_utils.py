"""Test helper functions for the label evaluation tool."""

import pandas as pd
import pytest

from laurium.tools import evaluator_utils


def test_load_labels_returns_demo_data_when_no_path_given():
    """Test that a blank path falls back to the provided demo data."""
    demo = evaluator_utils.create_demo_human_labels()

    data = evaluator_utils.load_labels(
        path=None, demo=demo, label_column="label"
    )

    pd.testing.assert_frame_equal(data, demo)


def test_load_labels_reads_csv(tmp_path):
    """Test that a CSV path is read directly."""
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame({"record_id": [1, 2], "label": ["a", "b"]}).to_csv(
        csv_path, index=False
    )

    data = evaluator_utils.load_labels(
        path=str(csv_path),
        demo=evaluator_utils.create_demo_human_labels(),
        label_column="label",
    )

    assert list(data["label"]) == ["a", "b"]


def test_load_labels_missing_label_column_raises():
    """Test that a missing label column raises a ValueError."""
    demo = pd.DataFrame({"record_id": [1], "label": ["a"]})

    with pytest.raises(ValueError, match="not found"):
        evaluator_utils.load_labels(
            path=None, demo=demo, label_column="missing_column"
        )


def test_merge_labels_joins_on_id_and_renames_columns():
    """Test that human/model labels are joined and renamed for comparison."""
    human = pd.DataFrame({"record_id": [1, 2, 3], "label": ["a", "b", "c"]})
    model = pd.DataFrame({"record_id": [1, 2, 3], "label": ["a", "c", "c"]})

    merged = evaluator_utils.merge_labels(
        human, model, "record_id", "label", "label"
    )

    assert list(merged.columns) == [
        "record_id",
        "actual_label",
        "predicted_label",
    ]
    assert list(merged["actual_label"]) == ["a", "b", "c"]
    assert list(merged["predicted_label"]) == ["a", "c", "c"]


def test_merge_labels_excludes_unmatched_and_missing_rows():
    """Test that unmatched ids and missing labels are dropped."""
    human = pd.DataFrame({"record_id": [1, 2, 3], "label": ["a", None, "c"]})
    model = pd.DataFrame({"record_id": [1, 2], "label": ["a", "b"]})

    merged = evaluator_utils.merge_labels(
        human, model, "record_id", "label", "label"
    )

    assert list(merged["record_id"]) == [1]


def test_merge_labels_missing_column_raises():
    """Test that a missing id or label column raises a ValueError."""
    human = pd.DataFrame({"record_id": [1], "label": ["a"]})
    model = pd.DataFrame({"record_id": [1]})

    with pytest.raises(ValueError, match="Model data is missing"):
        evaluator_utils.merge_labels(
            human, model, "record_id", "label", "label"
        )


def test_merge_labels_preserves_auxiliary_columns():
    """Test that free text and explanation columns survive the merge."""
    human = pd.DataFrame(
        {
            "record_id": [1, 2],
            "text": ["first example", "second example"],
            "label": ["a", "b"],
        }
    )
    model = pd.DataFrame(
        {
            "record_id": [1, 2],
            "label": ["a", "a"],
            "explanation": ["looks like a", "could be a or b"],
        }
    )

    merged = evaluator_utils.merge_labels(
        human, model, "record_id", "label", "label"
    )

    assert list(merged["text"]) == ["first example", "second example"]
    assert list(merged["explanation"]) == [
        "looks like a",
        "could be a or b",
    ]


def test_merge_labels_suffixes_colliding_auxiliary_columns():
    """Test that clashing non-label column names are suffixed, not dropped."""
    human = pd.DataFrame(
        {"record_id": [1], "label": ["a"], "notes": ["human note"]}
    )
    model = pd.DataFrame(
        {"record_id": [1], "label": ["a"], "notes": ["model note"]}
    )

    merged = evaluator_utils.merge_labels(
        human, model, "record_id", "label", "label"
    )

    assert merged["notes_human"].iloc[0] == "human note"
    assert merged["notes_model"].iloc[0] == "model note"


def test_summarize_match_counts():
    """Test that dataset and match counts are reported correctly."""
    human = pd.DataFrame({"record_id": [1, 2, 3], "label": ["a", "b", "c"]})
    model = pd.DataFrame({"record_id": [1, 2], "label": ["a", "b"]})
    merged = evaluator_utils.merge_labels(
        human, model, "record_id", "label", "label"
    )

    counts = evaluator_utils.summarize_match_counts(human, model, merged)

    assert counts == {"human_rows": 3, "model_rows": 2, "matched_rows": 2}


def test_build_confusion_counts_includes_zero_count_combinations():
    """Test that the confusion matrix includes all label combinations."""
    merged = pd.DataFrame(
        {
            "actual_label": ["a", "a", "b"],
            "predicted_label": ["a", "b", "b"],
        }
    )

    counts = evaluator_utils.build_confusion_counts(merged)

    lookup = {
        (row.actual_label, row.predicted_label): row.count
        for row in counts.itertuples()
    }
    assert lookup[("a", "a")] == 1
    assert lookup[("a", "b")] == 1
    assert lookup[("b", "a")] == 0
    assert lookup[("b", "b")] == 1
    assert len(counts) == 4


@pytest.fixture(name="example_data")
def example_data_fixture():
    """Build a small merged dataframe with text and explanation columns."""
    return pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "text": ["first", "second", "third"],
            "actual_label": ["a", "a", "b"],
            "predicted_label": ["a", "b", "b"],
            "explanation": ["exp1", "exp2", "exp3"],
        }
    )


def test_filter_examples_by_cell_returns_all_when_no_selection(example_data):
    """Test that no selection returns every example unchanged."""
    result = evaluator_utils.filter_examples_by_cell(example_data, None, None)

    pd.testing.assert_frame_equal(result, example_data)


def test_filter_examples_by_cell_filters_to_matching_pair(example_data):
    """Test that a selected cell returns only matching examples."""
    result = evaluator_utils.filter_examples_by_cell(example_data, "a", "b")

    assert list(result["record_id"]) == [2]


def test_filter_examples_by_cell_empty_for_zero_count_cell(example_data):
    """Test that a cell with no matches returns an empty dataframe."""
    result = evaluator_utils.filter_examples_by_cell(example_data, "b", "a")

    assert result.empty


def test_select_display_columns_includes_available_optional_columns(
    example_data,
):
    """Test that id, text, explanation and label columns are selected."""
    result = evaluator_utils.select_display_columns(
        example_data, "record_id", "text", "explanation"
    )

    assert list(result.columns) == [
        "record_id",
        "text",
        "explanation",
        "actual_label",
        "predicted_label",
    ]


def test_select_display_columns_skips_missing_optional_columns(example_data):
    """Test that a missing optional column is silently omitted."""
    result = evaluator_utils.select_display_columns(
        example_data, "record_id", "text", "missing_column"
    )

    assert list(result.columns) == [
        "record_id",
        "text",
        "actual_label",
        "predicted_label",
    ]
