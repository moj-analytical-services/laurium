"""Submodule for computing metrics as required by fine_tuner."""

import evaluate
import numpy as np


def compute_metrics(
    eval_pred: tuple[np.ndarray, np.ndarray],
    metrics: list[str],
    config_name: str = "multiclass",
    average: str = "macro",
) -> dict:
    """Calculate metrics scores for model predictions.

    This function computes metrics scores between model predictions
    and ground truth labels.

    Parameters
    ----------
    eval_pred : tuple[np.ndarray, np.ndarray]
        A tuple containing (logits, labels) where:
        - logits: numpy array of shape (n_samples, n_classes) containing model
            predictions
        - labels: numpy array of shape (n_samples,) containing ground truth
            labels
    metrics : list[str]
        List of metrics for evaluation
    config_name : str, default="mutliclass"
        Determine configuration for metrics
    average: str, default="macro"
        The averaging method to use for calculating metrics scores.
        - "macro": Calculate metrics for each label and find their unweighted
            mean.
        - "micro": Calculate metrics globally by counting true positives, false
            negatives, and false positives.
        - "weighted": Calculate metrics for each label and find their weighted
            mean by support.
        - "none": Calculate metrics for each label and return the metric for
            every label.

    Returns
    -------
    dict
        A dictionary with the computed metrics scores.

    Notes
    -----
    The function uses the HuggingFace evaluate library to compute metrics
    scores with macro-averaging by default, which calculates metrics for each
    label and finds their unweighted mean.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    results = {}
    for metric in metrics:
        if metric == "accuracy":
            m = evaluate.load(metric)
            result = m.compute(predictions=predictions, references=labels)
        else:
            m = evaluate.load(metric, config_name=config_name)
            result = m.compute(
                predictions=predictions, references=labels, average=average
            )
        results.update(result)

    return results
