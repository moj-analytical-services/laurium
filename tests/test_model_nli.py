"""Test NLI model functions."""

import pytest
import torch

from laurium.encoder_models import preprocess


@pytest.fixture(name="hypothesis")
def hypothesis():
    """Defining default hypothesis to be used in testing suite."""  # noqa: D401
    return "This person has children."


@pytest.mark.parametrize(
    ["cluster_id_list", "expected_output"],
    [
        (
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [
                "This person has children.",
                "This person has children.",
                "This person has children.",
                "This person has children.",
                "This person has children.",
                "This person has children.",
                "This person has children.",
                "This person has children.",
                "This person has children.",
                "This person has children.",
            ],
        )
    ],
)
def test_ungendered_hypothesis(
    cluster_id_list: list[int], hypothesis: str, expected_output: list[str]
):
    """Test hypothesis function on ungendered hypothesis.

    Parameters
    ----------
        cluster_id_list: list[str]
            List of dummy cluster ids.
        hypothesis : str
            The base hypothesis to be tested.
        expected_output : list[str]
            A list of expected hypotheses.
    """
    preprocessor = preprocess.Preprocessor(
        pretrained_checkpoint="cross-encoder/nli-deberta-v3-small",
        context_window_size=2,
        hypothesis=hypothesis,
        gendered_hypotheses=False,
    )

    hypotheses = preprocessor.get_hypotheses(cluster_id_list)
    assert hypotheses == expected_output


@pytest.mark.parametrize(
    ["hypothesis_dict", "genders_dict", "expected_output"],
    [
        (
            {
                "F": "She has children.",
                "M": "He has children.",
                "": "This person has children.",
            },
            {0: "F", 1: "M", 2: "", 3: "F"},
            [
                "She has children.",
                "He has children.",
                "This person has children.",
                "She has children.",
            ],
        )
    ],
)
def test_gendered_hypothesis(
    hypothesis: str,
    hypothesis_dict: dict[str, str],
    genders_dict: dict[int, str],
    expected_output: list[str],
):
    """
    Test hypothesis function on gendered hypothesis.

    Parameters
    ----------
    hypothesis : str
        The base hypothesis to be tested.
    hypothesis_dict : dict[str, str]
        A dictonary of genders and the corresponding hypotheses
    genders_dict : dict[int, str]
        A dictionary of genders keyed by cluster id.
    expected_output : list[str]
        A list of expected hypotheses.
    """
    preprocessor = preprocess.Preprocessor(
        "cross-encoder/nli-deberta-v3-small",
        context_window_size=2,
        hypothesis=hypothesis,
        gendered_hypotheses=True,
        local_files_only=False,
    )

    hypotheses = preprocessor.get_hypotheses(
        list(genders_dict), hypothesis_dict, genders_dict
    )
    assert hypotheses == expected_output


@pytest.mark.parametrize(
    ["contexts", "hypotheses", "expected_output"],
    [
        (
            ["A short context."],
            ["A short hypothesis."],
            [
                (
                    "[CLS] A short context.[SEP] A short hypothesis.[SEP][PAD]"
                    "[PAD][PAD][PAD][PAD][PAD][PAD][PAD][PAD]"
                )
            ],
        ),
        (
            ["A short context which should be truncated." * 10],
            [
                (
                    "A long hypothesis which should take up most of the "
                    "context window."
                )
            ],
            [
                (
                    "[CLS] A short context which[SEP] A long hypothesis which "
                    "should take up most of the context window.[SEP]"
                )
            ],
        ),
    ],
)
def test_tokenize_text_pairs(
    hypothesis: str,
    contexts: list[str],
    hypotheses: list[str],
    expected_output: list[str],
):
    """
    Test tokenizer function.

    Parameters
    ----------
    hypothesis : str
        The base hypothesis to be tested.
    contexts : list[str]
        A list of sentence contexts.
    hypotheses : list[str]
        A list of hypotheses.
    expected_output : list[str]
        The tokenizer output, truncating the context where necessary.
    """
    # create instance of class
    preprocessor = preprocess.Preprocessor(
        pretrained_checkpoint="cross-encoder/nli-deberta-v3-small",
        hypothesis=hypothesis,
        max_length=20,
        local_files_only=False,
    )

    output = preprocessor.tokenize_text_pairs(contexts, hypotheses)
    # decode the output to check against natural language rather than token ids
    decoded = preprocessor.tokenizer.batch_decode(output["input_ids"])
    assert decoded == expected_output


@pytest.mark.parametrize(
    "tokenizer_output",
    [
        {
            "input_ids": torch.as_tensor([[0, 1, 3], [2, 4, 10]]),
            "token_type_ids": torch.as_tensor([[0, 0, 0], [0, 0, 1]]),
            "attention_mask": torch.as_tensor([[1, 1, 1], [1, 1, 1]]),
        }
    ],
)
def test_tokenizer_outputs_to_tensor_dataset(
    hypothesis: str,
    tokenizer_output: dict[str, torch.Tensor],
):
    """
    Test conversion from tokenizer outputs to tensor dataset.

    Parameters
    ----------
    hypothesis : str
        The base hypothesis to be tested.
    tokenizer_output : dict[str, torch.Tensor]
        The tokenizer output to be tested.
    """
    preprocessor = preprocess.Preprocessor(
        pretrained_checkpoint="cross-encoder/nli-deberta-v3-small",
        hypothesis=hypothesis,
        local_files_only=False,
    )

    tds = preprocessor.tokenizer_outputs_to_tensor_dataset(tokenizer_output)

    assert torch.equal(tds.tensors[0], tokenizer_output["input_ids"])
    assert torch.equal(tds.tensors[1], tokenizer_output["token_type_ids"])
    assert torch.equal(tds.tensors[2], tokenizer_output["attention_mask"])


@pytest.mark.parametrize(
    ["column_1", "column_2", "batched_col1", "batched_col2"],
    [
        (
            torch.as_tensor([[0, 1, 2], [2, 3, 4], [2, 3, 4]]),
            torch.as_tensor([[9, 5, 7], [8, 3, 3], [9, 5, 7]]),
            [
                torch.as_tensor([[0, 1, 2], [2, 3, 4]]),
                torch.as_tensor([[2, 3, 4]]),
            ],
            [
                torch.as_tensor([[9, 5, 7], [8, 3, 3]]),
                torch.as_tensor([[9, 5, 7]]),
            ],
        )
    ],
)
def test_tensor_dataset_to_dataloader(
    hypothesis: str,
    column_1: torch.Tensor,
    column_2: torch.Tensor,
    batched_col1: list[torch.Tensor],
    batched_col2: list[torch.Tensor],
):
    """
    Test conversion from tensor dataset to pytorch dataloader.

    Parameters
    ----------
    hypothesis : str
        The base hypothesis to be tested.
    column_1 : torch.Tensor
        The first column of the dataset.
    column_2 : torch.Tensor
        The second column of the dataset.
    batched_col1 : list[torch.Tensor]
        A list with batches of the first column of the dataset.
    batched_col2 : list[torch.Tensor]
        A list with batches of the second column of the dataset.
    """
    preprocessor = preprocess.Preprocessor(
        pretrained_checkpoint="cross-encoder/nli-deberta-v3-small",
        hypothesis=hypothesis,
        batch_size=2,
        local_files_only=False,
    )

    tds = torch.utils.data.TensorDataset(column_1, column_2)

    loader = preprocessor.tensor_dataset_to_dataloader(tds, shuffle=False)

    for (loader_col1, loader_col2), expected_col1, expected_col2 in zip(
        loader, batched_col1, batched_col2, strict=True
    ):
        assert torch.equal(loader_col1, expected_col1)
        assert torch.equal(loader_col2, expected_col2)
