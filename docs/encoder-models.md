## Encoder Model Setup
Laurium works with lightweight encoder models too:

### Lightweight Encoder Models
For running encoder-based models you need to optional encoder dependencies:

Please refer to guidance in the getting-started page for how to do so.

**Benefits:**

- Can run models without connection to cloud

- SetFit is useful when labelled data is limited

- Works offline

**Requirements:**

- GPU recommended for hyperparameter search and cross validation

- CPU can be used for SetFit-based applications


## Basic Usage

**Both Transformers and SetFit provide regular fine-tuning and hyperparameter search:**

- Below is an example showing regular fine-tuning for Transformers and SetFit

- Further examples illustrating hyperparameter search can be found in the how_to scripts in the notebooks directory

- We use optuna as the backend engine when running hyperparameter search


### Fine-tuning

Laurium specializes in structured data extraction from text. Here's how to
build a classification pipeline:


#### Using Transformers
```python
from datasets import load_dataset

from laurium.encoder_models.transformers.fine_tune import (
    DataConfig,
    FineTuner,
)

# Model configuration
classifier_model_init = {
    "pretrained_model_name_or_path": "microsoft/deberta-v3-small",
    "num_labels": 2,
    "local_files_only": False,
}

classifier_tokenizer_init = {
    "pretrained_model_name_or_path": "microsoft/deberta-v3-small",
    "use_fast": True,
}
# Tokenizer configuration
classifier_tokenizer_args = {
    "max_length": 128,
    "return_tensors": "pt",
    "padding": "max_length",
    "truncation": "longest_first",
}
# Training arguments for parameter tuning
classifier_training_args = {
    "output_dir": "./results",
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "num_train_epochs": 5,
    "weight_decay": 0.01,
    "save_strategy": "epoch",
    "report_to": "none",
    "eval_strategy": "epoch",
}

# Data configuration
classifier_data_config = DataConfig(text_column="text", label_column="label")
# Initialize fine-tuner
classifier_fine_tuner = FineTuner(
    metrics=["f1", "accuracy", "precision", "recall"],
    model_init=classifier_model_init,
    training_args=classifier_training_args,
    tokenizer_init=classifier_tokenizer_init,
    tokenizer_args=classifier_tokenizer_args,
    data_config=classifier_data_config,
)

# Prepare data and splits
classifier_tomatoes = load_dataset("rotten_tomatoes")
classifier_train_df = classifier_tomatoes["train"].to_pandas()
classifier_test_df = classifier_tomatoes["test"].to_pandas()

# Use a small subset for demonstration
small_train_df = classifier_train_df.sample(n=100, random_state=42)
small_test_df = classifier_test_df.sample(n=50, random_state=42)

# Fine-tune the model
trainer = classifier_fine_tuner.fine_tune_model(
    train_df=small_train_df, eval_df=small_test_df
)

# Evaluate the model
results = trainer.evaluate()
```


#### Using SetFit
```python
import setfit
from datasets import load_dataset

from laurium.encoder_models.setfit.setfit import SetFit
from laurium.components.setfit_eval_metrics import compute_metrics

# Prepare data and splits
classifier_tomatoes = load_dataset("rotten_tomatoes")
classifier_train_df = classifier_tomatoes["train"].to_pandas()
classifier_test_df = classifier_tomatoes["test"].to_pandas()

# Initialise model
setfit_model_init = {
        "pretrained_model_name_or_path": "sentence-transformers/all-MiniLM-L6-v2",
        "use_differentiable_head": False,
    }

# Training arguments for parameter tuning
setfit_training_args = {
    "sampling_strategy": "oversampling",
    "max_steps": -1,
    "end_to_end": False,
    "head_learning_rate": 1e-2,
    "body_learning_rate": [2e-5, 1e-5],
    "num_epochs": [1, 16],
    "logging_steps": 50,
    "eval_steps": 50,
    "save_steps": 100,
    "metric_for_best_model": "eval_embedding_loss",
}

# Initialize fine-tuner
setfit_fine_tuner = SetFit(
    metric=compute_metrics,
    model_init=setfit_model_init,
    training_args=setfit_training_args,
)

# Use a small test subset for demonstration
small_train_df = classifier_train_df.sample(n=100, random_state=42)
small_test_df = classifier_test_df.sample(n=50, random_state=42)

# Sample to 8 samples per class for training
sampled_train_df = (
    small_train_df.groupby("label", group_keys=False)
    .apply(
        lambda x: x.sample(n=min(len(x), 8), random_state=42)
    )
    .reset_index(drop=True)
)

# Fine-tune the model
trainer = setfit_fine_tuner.setfit_model_train(
    train_df=sampled_train_df, eval_df=small_test_df
)

# Evaluate the model
results = trainer.evaluate()
```


## Notebooks

The [`notebooks/` directory](
https://github.com/moj-analytical-services/laurium/tree/main/notebooks)
contains a combination of [Jupyter](https://jupyter.org/) and [marimo](
https://marimo.io/) notebooks for exploring Laurium. Transformer-based finetuner and SetFit
how to notebooks are placed in the marimo directory.

To run one of the marimo notebooks:

1. Clone the Laurium repo
2. Sync dependencies with [uv](https://docs.astral.sh/uv/) (`uv sync`)
3. Run the notebook of your choosing with the command

   ```bash
   uv run marimo run notebooks/[name of notebook].py
   ```
4. (**For more advanced users**) To get a deeper look at the code, you can
   open the notebook in "edit" mode, which allows you to view the code
   being run alongside the notebook itself.

   ```bash
   uv run marimo edit notebooks/[name of notebook].py
   ```

For more information about using marimo, check out [their documentation](
https://docs.marimo.io/getting_started/).


### Fine-tuning notebook

The [fine-tuning notebook](
https://github.com/moj-analytical-services/laurium/blob/main/notebooks/fine_tuner_howto.py)
illustrates how to condunct standard fine-tuning of transformer models as well as how to run
hyperparameter search for transformers-based models. This notebook is best run in marimo's edit
mode, allowing the user to view both the code and the output at the same time.

### SetFit notebook

The [setfit notebook](
https://github.com/moj-analytical-services/laurium/blob/main/notebooks/setfit_howto.py)
illustrates how to condunct standard fine-tuning of setfit models as well as how to run
hyperparameter search for setfit-based models. This notebook is best run in marimo's edit
mode, allowing the user to view both the code and the output at the same time.


## Recommended Models

### Transformers
- `microsoft/deberta-v3-small` - Best lightweight model
- `cross-encoder/nli-deberta-v3-small` - Best lightweight model for NLI

### SetFit
- `sentence-transformers/all-MiniLM-L6-v2` - Best lightweight sentence transformer
