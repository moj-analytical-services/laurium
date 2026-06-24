"""Test spaCy pipeline components."""

from typing import Optional

import pandas as pd
import pytest
import spacy

from laurium.components import extract_context


@pytest.fixture(name="extra_punct_chars")
def extra_punct_chars():
    """
    Extra punctuation characters for initializing `SpacyPipeline`.

    Returns
    -------
    list of str
        A list of additional punctuation characters to be considered during
        sentence segmentation.
    """
    return ["\n", "\v", "\f", "\r"]


@pytest.fixture(name="keywords")
def keywords():
    """
    Fixture to define keyword patterns for initializing `SpacyPipeline`.

    Returns
    -------
    list of dict
        A list of keyword patterns formatted according to spaCy's matcher
        specifications.
    """
    return [
        {
            "LEMMA": {
                "IN": [
                    "son",
                    "daughter",
                    "child",
                    "kid",
                    "stepson",
                    "stepdaughter",
                    "stepchild",
                ]
            }
        }
    ]


@pytest.mark.parametrize(
    "extra_punct_chars",
    [
        (None),
        (["extra_chars"]),
    ],
)
def test_spacy_pipeline_punct_chars(extra_punct_chars):
    """Test extra_punct_chars works when None and when defined."""
    base_chars = spacy.pipeline.sentencizer.Sentencizer.default_punct_chars
    output = extract_context.SpacyPipeline(
        [{"IS_PUNCT": True}],
        extra_punct_chars=extra_punct_chars,
    )

    if extra_punct_chars is None:
        extra_punct_chars = []
    assert set(output.punct_chars) == set(base_chars + extra_punct_chars)


@pytest.mark.parametrize(
    ["separator", "sentence_idx", "context_size", "expected_output"],
    [
        (".", 0, 1, "Sentence number 0. Sentence number 1."),
        (
            ".",
            1,
            1,
            "Sentence number 0. Sentence number 1. Sentence number 2.",
        ),
        (
            ".",
            19,
            4,
            (
                "Sentence number 15. Sentence number 16. Sentence number 17. "
                "Sentence number 18. Sentence number 19."
            ),
        ),
        (
            ".",
            22,
            4,
            # Some expected behavior when index exceeds the number of sentences
            "Sentence number 18. Sentence number 19.",
        ),
        (".", 24, 4, ""),
        (",", 2, 4, None),
        (";", 2, 4, None),
    ],
)
def test_generate_long_context_from_doc(
    keywords: list[dict[str, dict[str, list[str]]]],
    separator: str,
    sentence_idx: int,
    context_size: int,
    expected_output: Optional[str],
    extra_punct_chars: Optional[list[str]],
):
    """
    Test the `generate_long_context_from_doc` method of `SpacyPipeline`.

    Parameters
    ----------
    keywords : list of dict
        Keyword patterns used to initialize the `SpacyPipeline`.
    separator : str
        Separator to use when generating the test document.
    sentence_idx : int
        Index of the sentence for which context is to be generated.
    context_size : int
        Number of sentences to include as context.
    expected_output : str or None
        The expected output string. If `None`, the expected output is assumed
        to be a stripped version of the generated input.
    extra_punct_chars : list of str, optional
        Additional punctuation characters to consider when segmenting
        sentences. These characters are appended to spaCy's default punctuation
        characters.
    """
    # Create an instance of SpacyPipeline for testing
    spacy_pipeline = extract_context.SpacyPipeline(
        keywords=keywords, extra_punct_chars=extra_punct_chars
    )

    # Generate a long paragraph using the separator
    long_text = "".join(
        [f"Sentence number {i}{separator} " for i in range(20)]
    )
    doc = spacy_pipeline.spacy_model(long_text)

    if expected_output is None:
        expected_output = str(doc).strip()

    output = str(
        spacy_pipeline.generate_long_context_from_doc(
            doc, sentence_idx, context_size
        )
    )
    assert output == expected_output


@pytest.mark.parametrize(
    ["case_note", "expected_output"],
    [
        (
            ["This sentence contains no keyword."],
            ([], [], []),
        ),
        (
            ["This sentence contains the keyword children."],
            (
                [0],
                ["This sentence contains the keyword children."],
                ["children"],
            ),
        ),
        (
            ["This sentence contains the keyword children and daughter."],
            (
                [0],
                ["This sentence contains the keyword children and daughter."],
                ["children"],
            ),
        ),
        (
            ["Some dropped context. Some included context. Stepdaughters."],
            (
                [0],
                ["Some included context. Stepdaughters."],
                ["stepdaughters"],
            ),
        ),
        (
            ["Some dropped context. Some included context. Stepsons."],
            (
                [0],
                ["Some included context. Stepsons."],
                ["stepsons"],
            ),
        ),
        (
            ["Some dropped context. Some included context. Step-daughter."],
            (
                [0],
                ["Some included context. Step-daughter."],
                ["daughter"],
            ),
        ),
        (
            ["This sentence contains step-son."],
            (
                [0],
                ["This sentence contains step-son."],
                ["son"],
            ),
        ),
        (
            ["Some dropped context. Some included context. Step-sons."],
            (
                [0],
                ["Some included context. Step-sons."],
                ["sons"],
            ),
        ),
        pytest.param(
            ["Some dropped context. Some included context. Daughter-in-law."],
            ([], [], []),
            marks=pytest.mark.xfail(reason="Family edge case"),
        ),
        (
            ["Some dropped context. Some included context. Son-in-law."],
            ([], [], []),
        ),
        (
            ["This sentence contains the word grandaughter - not a keyword."],
            ([], [], []),
        ),
    ],
)
def test_get_keyword_mentions(
    keywords: list[dict[str, dict[str, list[str]]]],
    case_note: list[str],
    expected_output: tuple[list[int], list[str], list[str]],
    extra_punct_chars: Optional[list[str]],
    context_window_size: int = 2,
):
    """
    Test the `get_keyword_mentions` method of `SpacyPipeline`.

    Parameters
    ----------
    keywords : list of dict
        Keyword patterns used to initialize the `SpacyPipeline`.
    case_note : list[str]
        A list of case notes to be processed.
    expected_output : tuple
        A tuple containing lists of indices, contexts, and matched words
        expected from the pipeline.
    extra_punct_chars : list of str, optional
        Additional punctuation characters for sentence segmentation.
    context_window_size : int, optional
        Number of sentences to include as context. Default is `2`.
    """
    # Create an instance of SpacyPipeline for testing
    spacy_pipeline = extract_context.SpacyPipeline(
        keywords=keywords,
        context_window_size=context_window_size,
        extra_punct_chars=extra_punct_chars,
    )

    output = spacy_pipeline.get_keyword_mentions(case_note)
    assert output == expected_output


@pytest.mark.parametrize(
    ["chunk", "expected_output"],
    [
        (
            pd.DataFrame(
                {
                    "cluster_id": [0, 0, 0, 1, 1],
                    "note_text": [
                        "Mention of child.",
                        "No keyword.",
                        (
                            "Keyword: son. A sentence following the keyword. "
                            "A dropped sentence."
                        ),
                        "No keyword",
                        "No keyword",
                    ],
                }
            ),
            pd.DataFrame(
                {
                    "cluster_id": [0, 0],
                    "context": [
                        "Mention of child.",
                        "Keyword: son. A sentence following the keyword.",
                    ],
                    "matched_word": ["child", "son"],
                }
            ),
        )
    ],
)
def test_generate_df_context(
    keywords: list[dict[str, dict[str, list[str]]]],
    chunk: pd.DataFrame,
    extra_punct_chars: Optional[list[str]],
    expected_output: pd.DataFrame,
    context_window_size: int = 2,
):
    """
    Test the `generate_df_context` method of `SpacyPipeline`.

    Parameters
    ----------
    keywords : list of dict
        Keyword patterns used to initialize the `SpacyPipeline`.
    chunk : pandas.DataFrame
        DataFrame containing case notes with columns like 'cluster_id' and
        'note_text'.
    extra_punct_chars : list of str, optional
        Additional punctuation characters for sentence segmentation.
    expected_output : pandas.DataFrame
        The expected output DataFrame containing the context and matched
        keywords.
    context_window_size : int, optional
        Number of sentences to include as context. Default is `2`.
    """
    spacy_pipeline = extract_context.SpacyPipeline(
        keywords=keywords,
        extra_punct_chars=extra_punct_chars,
        context_window_size=context_window_size,
    )

    df_context = spacy_pipeline.generate_df_context(chunk)
    assert df_context.equals(expected_output)


@pytest.mark.parametrize(
    ["chunk", "context", "expected_output"],
    [
        (
            pd.DataFrame({"cluster_id": [0, 0, 0, 1, 1], "note_text": "Note"}),
            pd.DataFrame(
                {
                    "cluster_id": [0, 0],
                    "context": "Sentences",
                    "matched_word": "Words",
                }
            ),
            pd.DataFrame(
                {
                    "cluster_id": [0, 1],
                    "n_notes": [3, 2],
                    "n_mentions": [2.0, 0.0],
                }
            ),
        )
    ],
)
def test_generate_df_stats(
    keywords: list[dict[str, dict[str, list[str]]]],
    extra_punct_chars: Optional[list[str]],
    chunk: pd.DataFrame,
    context: pd.DataFrame,
    expected_output: pd.DataFrame,
):
    """
    Test the `generate_df_stats` method of `SpacyPipeline`.

    Parameters
    ----------
    keywords : list of dict
        Keyword patterns used to initialize the `SpacyPipeline`.
    extra_punct_chars : list of str, optional
        Additional punctuation characters for sentence segmentation.
    chunk : pandas.DataFrame
        Input DataFrame containing case notes with columns like 'cluster_id'
        and 'note_text'.
    context : pandas.DataFrame
        DataFrame containing contexts and matched words generated from the
        chunk.
    expected_output : pandas.DataFrame
        The expected statistics DataFrame with columns like 'cluster_id',
        'n_notes', and 'n_mentions'.
    """
    spacy_pipeline = extract_context.SpacyPipeline(
        keywords=keywords, extra_punct_chars=extra_punct_chars
    )

    df_stats = spacy_pipeline.generate_df_stats(chunk, context)
    assert df_stats.equals(expected_output)
