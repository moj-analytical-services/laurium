"""Test data cleaning and preprocessing functions."""

import pytest

from laurium.components import utils


@pytest.mark.parametrize(
    ["input", "expected_output"],
    [
        ("\n\nTest string.", "\nTest string."),
        ("\n\n\nTest string.", "\nTest string."),
        ("\n\n\nTest string.\n", "\nTest string.\n"),
        ("\n\n\nTest string.\n\n", "\nTest string.\n"),
        (
            "\n\n\nTest string.\n\nTest string.\n\n",
            "\nTest string.\nTest string.\n",
        ),
        ("\n\n", "\n"),
        ("", ""),
        ("\u2029\u2029Test string.", "\nTest string."),
        ("\u2029Test string.", "\nTest string."),
    ],
)
def test_regex_convert_vertical_whitespace(input: str, expected_output: str):
    """
    Test whitespace standardisation function.

    Parameters
    ----------
    input : str
        The input test string.
    expected_output : str
        The expected output, with white space reduced.
    """
    output = utils.regex_convert_vertical_whitespace(input)
    assert output == expected_output
