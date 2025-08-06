"""
nli.py module provides a PyTorch Lightning Module for NLI tasks.

It defines the `NLIModelPL` class, which leverages a pre-trained transformer
model to perform NLI by calculating the probability of entailment between pairs
of sentences. The model is built using Hugging Face Transformers and PyTorch
Lightning, facilitating easy integration into training and inference pipelines.

Classes
-------
NLIModelPL : pl.LightningModule
    A PyTorch Lightning Module for NLI tasks using a pre-trained transformer
    model.

Usage
-----
Instantiate the `NLIModelPL` class with the desired pre-trained model
checkpoint and use it to predict entailment probabilities for sentence pairs.

"""

from typing import Optional

import pytorch_lightning as pl
import torch
import torch.utils.data
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class NLIModelPL(pl.LightningModule):
    """
    Perform inference using a sequence classification model from Transformers.

    It calculates the probability of entailment between pairs of sentences.

    Parameters
    ----------
    pretrained_checkpoint : str, optional
        The path or identifier of the pre-trained model checkpoint. Default is
        "cross-encoder/nli-deberta-v3-small".
    idx_entailment : int, optional
        The index of the "entailment" label in the model's output logits. This
        may vary depending on the model. Default is 1.
    local_files_only : bool, optional
        Whether to only load models from local files and not attempt to
        download. Default is True.
    **kwargs
        Additional keyword arguments, which are passed to pl.LightningModule.

    Attributes
    ----------
    predict_outputs : list
        List to store prediction outputs during prediction.
    p_entail : torch.Tensor
        Tensor containing the probability of entailment for each input sample.
    """

    def __init__(
        self,
        pretrained_checkpoint: str = "cross-encoder/nli-deberta-v3-small",
        idx_entailment: int = 1,  # NB: order is model-dependent!
        local_files_only: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_checkpoint, local_files_only=local_files_only
        )
        self.idx_entailment = idx_entailment
        self.p_entail = None

    def on_predict_epoch_start(self):
        """
        Initialize self.predict_outputs (list) to hold prediction outputs.

        Notes
        -----
        This method is necessary due to changes in PyTorch Lightning v2.0
        regarding memory usage.  See
        https://github.com/Lightning-AI/pytorch-lightning/pull/16520 for more
        details.
        """
        super().on_predict_epoch_start()
        self.predict_outputs = []

        return

    def predict_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: Optional[int] = None,
    ) -> SequenceClassifierOutput:
        """
        Make predictions for one batch of data.

        Parameters
        ----------
        batch : tuple of torch.Tensor
            Batched data containing input_ids, token_type_ids, attention_mask.
        batch_idx : int, optional
            Index of current batch, defined for pipeline compatibility.
            Defaults to None.

        Returns
        -------
        outputs : transformers.modeling_outputs.SequenceClassifierOutput
            Model outputs, which includes the 'logits' key.
        """
        input_ids, token_type_ids, attention_mask = batch

        outputs = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        self.predict_outputs.append(outputs)
        return outputs

    def get_p_entail(self) -> torch.Tensor:
        """
        Calculate probability of entailment using self.predict_outputs.

        Returns
        -------
        torch.Tensor
            Probability of entailment for each input sample.
        """
        # Get logits from all batches (self.predict_outputs is list[dict])
        logits = torch.cat([x["logits"] for x in self.predict_outputs])

        # Softmax to convert logits to probabilities; keep "entailment" column
        self.p_entail = torch.softmax(logits, dim=-1)[:, self.idx_entailment]

        # Clear this in case of running prediction on multiple dataloaders
        self.predict_outputs.clear()

        return self.p_entail

    def predict(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Calculate the probability of entailment given premise and hypothesis.

        Parameters
        ----------
        dataloader : torch.utils.data.DataLoader
            Generator containing batched data.

        Returns
        -------
        torch.Tensor
            Probability of entailment for each input sample.
        """
        self.on_predict_epoch_start()

        with torch.no_grad():
            for batch in dataloader:
                self.predict_step(batch)

        p_entail = self.get_p_entail()

        return p_entail
