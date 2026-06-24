"""Test SQL data loading functions."""

import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pandasql
import pytest
from pytest_mock.plugin import MockerFixture

from laurium.components.load import data_loader


@pytest.fixture
def free_text_df() -> Generator[pd.DataFrame, None, None]:
    """Fixture data frame of free text.

    Yields
    ------
    pandas.DataFrame
        A data frame of free text.
    """
    df = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "free_text": [
                "Alice was helpful today.",
                "Bob was disruptive.",
                "Charlie is concerned about his son's health.",
            ],
        }
    )

    yield df


@pytest.fixture
def query_template() -> Generator[Path, None, None]:
    """Fixture query template file.

    Yields
    ------
    pathlib.Path
        Filepath to the template query file.
    """
    template = """
    SELECT *
    FROM free_text

    LIMIT {{ offset }},{{ chunk_size }}
    """

    with tempfile.TemporaryDirectory() as tempdir:
        template_path = Path(tempdir) / "template.sql"
        with open(template_path, "w") as f:
            f.write(template)

        yield template_path


@pytest.fixture
def mocked_db(mocker: MockerFixture, free_text_df: pd.DataFrame) -> None:
    """Fixture to mock calls to AWS Athena.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        pytest-mock plugin
    free_text_df : pandas.DataFrame
        The data frame fixture to be queried in the mocked call
    """

    def patched_read_sql(query, **kwargs):
        return pandasql.sqldf(query, {"free_text": free_text_df})

    mocker.patch("pydbtools.read_sql_query", side_effect=patched_read_sql)


def test_data_loader(
    mocked_db: None,
    free_text_df: pd.DataFrame,
    query_template: Path,
) -> None:
    """Test `data_loader`.

    Parameters
    ----------
    mocked_db : None
        A fixture that will mock calls to Athena.
    free_text_df : pd.DataFrame
        The ground truth database to compare with data_loader.
    query_template : Path
        The path to the SQL query template to use.
    """
    loader = data_loader(1, query_template)
    for actual, (_, expected) in zip(
        loader, free_text_df.iterrows(), strict=True
    ):
        pd.testing.assert_series_equal(
            actual.loc[0],  # type: ignore[arg-type]
            expected,
            check_names=False,
        )
