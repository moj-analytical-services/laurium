"""
Provides a class for preparing data for NLI models.

The `Preprocessor` class handles tokenization, data loading, and other
preprocessing tasks specific to PyTorch and Hugging Face Transformers. It
includes methods for tokenizing text pairs, generating hypotheses (optionally
gendered), and creating `DataLoader` objects for training and inference.

Classes
-------
Preprocessor
    A class for preprocessing tasks, including tokenization and data loading
    for NLI models.

Usage
-----
Instantiate the `Preprocessor` class with the desired parameters and use its
methods to preprocess your dataset for NLI tasks.

"""

from typing import Optional

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer


class Preprocessor:
    """
    Preprocessor class for PyTorch/Transformers-specific preprocessing tasks.

    This class provides methods for tokenizing text pairs for Natural Language
    Inference (NLI) tasks using a specified tokenizer, and prepares data
    loaders for use with PyTorch models.

    Parameters
    ----------
    pretrained_checkpoint : str
        Path or identifier of the pre-trained NLI model.
    hypothesis : str, optional
        Default hypothesis to test against the premise.
    gendered_hypotheses : bool, optional
        Whether to use gender when generating hypotheses. Default is False.
    context_window_size : int, optional
        Number of sentences to use as premise for each input. Default is 1.
    max_length : int, optional
        Maximum number of tokens above which inputs will be truncated.
        Default is 128.
    batch_size : int, optional
        Batch size for the transformer model. Default is 32.
    local_files_only : bool, optional
        Whether to only load models from local files. Default is True.
    """  # noqa: D205

    def __init__(
        self,
        pretrained_checkpoint: str,
        hypothesis: str,
        gendered_hypotheses: bool = False,
        context_window_size: int = 1,
        max_length: int = 128,
        batch_size: int = 32,
        local_files_only: bool = False,
    ):
        self.pretrained_checkpoint = pretrained_checkpoint
        self.context_window_size = context_window_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.hypothesis = hypothesis
        self.gendered_hypotheses = gendered_hypotheses

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_checkpoint,
            local_files_only=local_files_only,
        )

    def get_hypotheses(
        self,
        cluster_ids: list[int],
        hypothesis_dict: Optional[dict[str, str]] = None,
        genders_dict: Optional[dict[int, str]] = None,
    ) -> list[str]:
        """
        Generate (optionally gendered) hypotheses for NLI input.

        Parameters
        ----------
        cluster_ids : list of int
            List of cluster IDs, one per input context/premise (likely with
            repeats).
        hypothesis_dict : dict, optional
            Dictionary mapping genders to hypothesis.
        genders_dict : dict, optional
            Dictionary mapping cluster IDs to gender labels ('M' or 'F').

        Returns
        -------
        list of str
            List of hypotheses.

        Raises
        ------
        ValueError
            If `gendered_hypotheses` is True and `genders_dict` is None.
        ValueError
            If `hypothesis_dict` is None when gendered hypotheses are expected.
        KeyError
            If a cluster ID is not in `genders_dict` or a gender is not in
            `hypothesis_dict`.
        """
        if not self.gendered_hypotheses:
            return [self.hypothesis] * len(cluster_ids)
        else:
            if genders_dict is None:
                raise ValueError(
                    "genders_dict must be provided when gendered_hypotheses "
                    "is True"
                )

            if hypothesis_dict is None:
                raise ValueError(
                    "hypothesis_dict must be provided when "
                    "gendered_hypotheses is True"
                )

            try:
                hypotheses = [
                    hypothesis_dict[genders_dict[cid]] for cid in cluster_ids
                ]

            except KeyError as exc:
                raise KeyError(
                    f"Missing cluster id or gender for cluster id: {exc}"
                ) from exc

            return hypotheses

    def tokenize_text_pairs(
        self, contexts: list[str], hypotheses: list[str]
    ) -> dict[str, torch.Tensor]:
        """
        Given contexts and hypotheses, returns tokenized data as tensors.

        Parameters
        ----------
        contexts : list of str
            List of contexts (sentences containing child mention).
        hypotheses : list of str
            List of hypotheses to test, e.g., 'This person has children.'

        Returns
        -------
        dict
            Tokenizer outputs with keys 'input_ids', 'token_type_ids',
            'attention_mask'; values are PyTorch tensors with dimensions
            (batch_size, max_length).
        """
        tokenizer_outputs = self.tokenizer(
            contexts,
            hypotheses,
            truncation="only_first",
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return tokenizer_outputs

    def tokenizer_outputs_to_tensor_dataset(
        self, tokenizer_outputs: dict[str, torch.Tensor]
    ) -> TensorDataset:
        """
        Wrap tokenized outputs in a PyTorch TensorDataset.

        Parameters
        ----------
        tokenizer_outputs : dict
            Dictionary containing 'input_ids', 'token_type_ids',
            'attention_mask'.

        Returns
        -------
        torch.utils.data.TensorDataset
            Dataset for use with PyTorch DataLoaders.
        """
        input_ids, token_type_ids, attention_mask = tokenizer_outputs.values()
        dataset = TensorDataset(input_ids, token_type_ids, attention_mask)

        return dataset

    def tensor_dataset_to_dataloader(
        self, dataset: TensorDataset, shuffle: bool = False
    ) -> DataLoader:
        """
        Wrap a TensorDataset in a PyTorch DataLoader.

        Parameters
        ----------
        dataset : torch.utils.data.TensorDataset
            Dataset containing 'input_ids', 'token_type_ids', 'attention_mask'.
        shuffle : bool, optional
            Whether to shuffle the dataset. Useful for training.
            Default is False.

        Returns
        -------
        torch.utils.data.DataLoader
            Iterable DataLoader containing batched inputs.
        """
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def text_pair_to_dataloader(
        self, contexts: list[str], hypotheses: list[str]
    ) -> DataLoader:
        """
        Given contexts and hypotheses, returns a PyTorch DataLoader.

        Parameters
        ----------
        contexts : list of str
            List of context/premise sentences for NLI.
        hypotheses : list of str
            List of hypotheses for NLI.

        Returns
        -------
        torch.utils.data.DataLoader
            Iterable DataLoader where each sample contains 'input_ids',
            'token_type_ids', and 'attention_mask'.
        """
        tokenizer_outputs = self.tokenize_text_pairs(contexts, hypotheses)
        dataset = self.tokenizer_outputs_to_tensor_dataset(tokenizer_outputs)
        dataloader = self.tensor_dataset_to_dataloader(dataset)

        return dataloader

    def df_context_to_dataloader(
        self, df_context: pd.DataFrame, genders_dict: dict[int, str]
    ) -> DataLoader:
        """
        Given contexts and a genders dictionary, returns a PyTorch DataLoader.

        Parameters
        ----------
        df_context : pandas.DataFrame
            DataFrame containing columns 'cluster_id' and 'context'.
        genders_dict : dict
            Dictionary with cluster IDs (int) as keys and genders
            (str, 'F'/'M') as values.

        Returns
        -------
        torch.utils.data.DataLoader
            Iterable DataLoader where each sample contains 'input_ids',
            'token_type_ids', and 'attention_mask'.
        """
        cluster_ids = df_context["cluster_id"].tolist()
        contexts = df_context["context"].tolist()
        hypotheses = self.get_hypotheses(
            cluster_ids=cluster_ids, genders_dict=genders_dict
        )

        return self.text_pair_to_dataloader(contexts, hypotheses)
