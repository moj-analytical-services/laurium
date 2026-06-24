"""Compute metrics for set fit trainer."""

import evaluate


def compute_metrics(y_pred, y_test, metrics_to_compute=None, average=None):
    """
    Compute custom metrics for SetFit Trainer evaluation.

    Args:
        eval_pred: tuple of (logits, labels)
        metrics_to_compute: list of metric names to compute (default: common metrics)
            Options: "accuracy", "precision", "recall", "f1", "roc_auc"

    Returns
    -------
        dict: Dictionary containing the computed metrics
    """
    if metrics_to_compute is None:
        metrics_to_compute = ["accuracy", "precision", "recall", "f1"]

    if average is None:
        average = "macro"

    if y_pred.ndim > 1:
        y_pred = y_pred.argmax(axis=1)

    results = {}

    for metric in metrics_to_compute:
        if metric == "accuracy":
            m = evaluate.load(metric)
            result = m.compute(predictions=y_pred, references=y_test)
        else:
            m = evaluate.load(metric)
            result = m.compute(
                predictions=y_pred, references=y_test, average=average
            )
        results.update(result)

    return results
