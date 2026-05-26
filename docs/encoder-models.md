## What are Encoder models
Encoder models are typically used for reading and understanding text.

Unlike decoder-only models such as GPT, encoder models focus on learning the
contextual meaning of text rather than generating new text.

### How do they work

Consider the following sentence for example: "The bank near the river flooded"

Encoder models read the entire passage to understand the context, and hence can
deduce that the word "bank" here refers to a river bank, rather than a financial
institution.

This ability to understand words using surrounding context is one of the key
strengths of encoder architectures.

**Encoder models are useful when you want to understand rather than generate text**

### Applications of Encoder Models

#### Binary Classification

This is where text is classified into one of two labels.

Examples include:
- Spam Detection
- Sentiment Analysis

#### Multi-Class Classification

This is where text is classified into one of multiple labels

Examples include:
- Topic Classification
- Document Type classification

#### Natural Language Inference (NLI)

This is where a hypothesis can either agree (entailment), disagree (contradiction),
or have no relation (neutral) to a premise

Examples include:
- Entailment Detection
- Case Law Similarity

## Encoder Model Architectures

Laurium consists of both Transformers and SetFit encoder architectures

### Transformers Architecture

#### High-level Workflow

The diagram below depicts the high-level transformers workflow.

```
Input Text
    ↓
Tokenizer
    ↓
Transformer Encoder
    ↓
Classification Head
    ↓
Predicted Label
```

1. The input text is first passed to a tokenizer, which converts the text into numerical token representations.

2. These tokens are then processed by the transformer encoder, which learns contextual meaning through self-attention mechanisms.

3. Finally, the classification head maps the learned representations to task-specific predictions.

#### Fine-tuning Transformer Models

Fine-tuning is the process of training a pre-trained transformer model on your data. It has two steps, a forward pass,
and a backwards pass.

##### Forward Pass

The workflow for the forward pass is shown. During the forward pass the loss is calculated between predicted and actual labels.

```
Input
  ↓
Encoder
  ↓
Classifier
  ↓
Loss
```

##### Backward Pass

During the backward pass, gradients are propagated backward from the loss through the classification head and transformer encoder
layers, allowing the model weights to be updated to reduce prediction error.

```
Loss
  ↑
Classifier gradients
  ↑
Encoder gradients
  ↑
Embedding gradients
```

### SetFit Architecture

SetFit stands for Sentence Transformer Fine-Tuning. It is a few-shot classification technique designed for scenarios where labelled data is limited.

SetFit uses contrastive learning to construct positive and negative sentence pairs from labelled examples. This allows the embedding model to learn
semantically meaningful relationships.

Following this a lightweight classifier (the default being Logistic Regression) is trained on top of those embeddings for downstream classification tasks.


#### High-level Workflow

```
Input Text
    ↓
Sentence Transformer Encoder
    ↓
Sentence Embeddings
    ↓
Contrastive Learning
    ↓
Lightweight Classifier
    ↓
Predicted Label
```

#### Contrastive Learning

Contrastive learning teaches a model which examples should be considered semantically similar and which should be considered different.

Example:
```
Sentence A:
"The service was excellent."

Sentence B:
"The customer experience was great."

Sentence C:
"The product arrived damaged."
```

The model learns:
```
A ↔ B  → close together
A ↔ C  → far apart
```
in vector space.

#### SetFit Fine-tuning

The default workflow is shown below. In it's default state SetFit employs a detatched Logistic Regression
classifier head on top of the embedding model.

```
Labelled Sentences
        ↓
Positive & Negative Pair Generation
        ↓
Sentence Transformer Encoder
        ↓
Sentence Embeddings
        ↓
Contrastive Loss Calculation
        ↓
Backpropagation
        ↓
Updated Encoder Weights
```

The sentence-transformer encoder generates embeddings for each sentence pair, and a contrastive loss function measures
how semantically close or distant those embeddings should be. Backpropagation is then used to update the encoder weights,
improving the quality of the learned embedding space.

### SetFit vs Traditional Transformer Fine-Tuning

Traditional transformer fine-tuning and SetFit are both approaches for adapting pretrained encoder models to downstream NLP tasks such as classification and NLI. However, they differ in how training is performed and the types of problems they are best suited for.

#### Traditional Transformer Fine-Tuning

Traditional transformer fine-tuning trains a transformer encoder together with a classification head using supervised learning.

#### SetFit

SetFit (Sentence Transformer Fine-Tuning) uses contrastive learning to optimise sentence embeddings before training a lightweight classifier.

#### Summary Comparison

| Feature | Transformer Fine-Tuning | SetFit |
|---|---|---|
| Training approach | Classification-first | Embedding-first |
| Primary learning method | Supervised classification | Contrastive learning |
| Best for | Medium/Large datasets | Small/Few-shot datasets |
| Training speed | Slower | Faster |
| Computational cost | Higher | Lower |
| Pipeline complexity | Higher | Lower |
| Flexibility | High | Moderate |
| Embedding quality | Good | Strong semantic embeddings |



### Recommended Encoder Models

#### Transformers
- `microsoft/deberta-v3-small` - Best lightweight model
- `cross-encoder/nli-deberta-v3-small` - Best lightweight model for NLI

#### SetFit
- `sentence-transformers/all-MiniLM-L6-v2` - Best lightweight sentence transformer

## Encoder Model Evaluation and Optimisation

### Hyperparameter Search

Hyperparameter search is the process of systematically evaluating different training configurations to identify the combination of parameters that produces the best model performance.

Unlike model weights, which are learned during training, hyperparameters are predefined values that control the training process itself. Common hyperparameters include learning rate, batch size, number of training epochs, weight decay, and warmup steps.

During hyperparameter search, multiple training runs are performed using different parameter combinations, and each model is evaluated using a chosen metric such as accuracy, F1-score, precision, or recall. The best-performing configuration can then be selected for final training.

Hyperparameter optimisation is important because model performance can vary significantly depending on the selected training configuration.


### Cross Validation

Cross-validation is a model evaluation technique used to assess how well a model generalises to unseen data.

Rather than training and evaluating on a single train-validation split, the dataset is divided into multiple subsets, or folds. The model is then trained multiple times, using a different fold as the validation set during each iteration while the remaining folds are used for training.

A common approach is k-fold cross-validation:

```
Fold 1 → Validation
Fold 2-5 → Training

Fold 2 → Validation
Fold 1,3,4,5 → Training
```

This process produces multiple evaluation scores, which are typically averaged to obtain a more reliable estimate of model performance.

Cross-validation helps reduce the risk of overfitting to a particular validation split and provides a more robust assessment of how the model is expected to perform on unseen data.


## Encoder How-to Notebooks

The `notebooks/` [directory](
https://github.com/moj-analytical-services/laurium/tree/main/notebooks) contains a combination of [Jupyter](https://jupyter.org/) and [marimo](
https://marimo.io/) notebooks for exploring Laurium. Transformer-based finetuner and SetFit
how to notebooks are placed in the marimo directory.


To run one of the marimo notebooks:

Clone the Laurium repo

Sync dependencies with uv (uv sync --extra encoder -extra notebooks)

Run the notebook of your choosing with the command

uv run marimo run notebooks/[name of notebook].py
(For more advanced users) To get a deeper look at the code, you can open the notebook in "edit" mode, which allows you to view the code being run alongside the notebook itself.

uv run marimo edit notebooks/[name of notebook].py
For more information about using marimo, check out their documentation.

**Both Notebooks contain examples of simple fine-tuning and hyperparameter search**
