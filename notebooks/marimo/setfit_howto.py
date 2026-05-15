import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    return


@app.cell
def _():
    import setfit
    from datasets import load_dataset

    from laurium.components.setfit_eval_metrics import compute_metrics
    from laurium.encoder_models.setfit.setfit import SetFit

    return SetFit, compute_metrics, load_dataset, setfit


@app.cell
def _(load_dataset):
    # Prepare data and splits
    classifier_tomatoes = load_dataset("rotten_tomatoes")
    classifier_train_df = classifier_tomatoes["train"].to_pandas()
    classifier_test_df = classifier_tomatoes["test"].to_pandas()
    return classifier_test_df, classifier_train_df


@app.cell
def _(SetFit, compute_metrics):

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
    return (setfit_fine_tuner,)


@app.cell
def _(classifier_test_df, classifier_train_df, setfit_fine_tuner):
    # Example 1: Regular fine-tuning
    print("=== Regular Fine-tuning ===")

    # Use a small test subset for demonstration
    small_train_df = classifier_train_df.sample(n=100, random_state=42)
    small_test_df = classifier_test_df.sample(n=50, random_state=42)

    # Sample to 8 samples per class for training
    sampled_train_df = (
        small_train_df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(n=min(len(x), 8), random_state=42))
        .reset_index(drop=True)
    )
    # Fine-tune the model
    trainer = setfit_fine_tuner.setfit_model_train(
        train_df=sampled_train_df, eval_df=small_test_df
    )

    # Evaluate the model
    results = trainer.evaluate()
    print(f"Regular fine-tuning results: {results}")
    return small_test_df, small_train_df


@app.cell
def _(setfit):
    print(setfit.__version__)
    return


@app.cell
def _(setfit_fine_tuner, small_test_df, small_train_df):
    print("\n=== Hyperparameter Search ===")

    # Create a trainer configured for hyperparameter search
    search_trainer = setfit_fine_tuner.create_trainer_for_search(
        train_df=small_train_df, eval_df=small_test_df
    )

    # Define hyperparameter space
    def hp_space(trial):
        return {
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            "per_device_train_batch_size": trial.suggest_categorical(
                "batch_size", [8, 16]
            ),
            "num_train_epochs": trial.suggest_int("epochs", 2, 4),
        }

    # Run hyperparameter search
    best_trial = search_trainer.hyperparameter_search(
        hp_space=hp_space,
        n_trials=3,  # Small number for demo
        direction="maximize",
    )

    print(f"Best hyperparameters: {best_trial.hyperparameters}")
    print(f"Best objective value: {best_trial.objective}")

    return


if __name__ == "__main__":
    app.run()
