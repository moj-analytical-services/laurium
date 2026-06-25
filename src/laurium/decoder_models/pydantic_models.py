"""Base models for LLM-based extraction."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model


def create_label_model(
    labels: list[str],
    config: ConfigDict | None = None,
) -> type[BaseModel]:
    """Generate a Pydantic model for a custom list of labels.

    Parameters
    ----------
    labels : list[str]
        The labels for the model.
    config : pydantic.ConfigDict, optional
        The Pydantic model configuration; default is {'extra': 'forbid'}.

    Returns
    -------
    type[BaseModel]
        A Pydantic model class with two fields:
            label - a string literal, with a value in `labels`
            explanation - a string explaining the reason for the label.
    """
    if config is None:
        config = ConfigDict(extra="forbid")

    label_field = Annotated[
        Literal[tuple(labels)],  # type: ignore
        Field(
            description=(
                f"The label assigned to the free text; one of {labels}."
            ),
        ),
    ]

    explanation_field = Annotated[
        str,
        Field(
            description="The explanation for why this label was chosen.",
        ),
    ]

    LabelModel = create_model(  #  noqa: N806
        "LabelModel",
        __config__=config,
        label=label_field,
        explanation=explanation_field,
    )

    return LabelModel


def make_dynamic_example_model(
    schema: dict[str, Any],
    descriptions: dict[str, str],
    model_name: str,
    docstring: str = "Schema for parsing LLM output",
) -> type[BaseModel]:
    """
    Build a Pydantic v2 model with required fields and descriptions.

    Parameters
    ----------
    schema : dict[str, type]
        Dictionary mapping field names to their types.
    descriptions : dict[str, str]
        Dictionary mapping field names to their descriptions.
    model_name : str
        Name for the dynamically created model.
    docstring : str, optional
        Docstring to attach to the created model, by default "Schema for
        parsing LLM output".

    Returns
    -------
    type[BaseModel]
        A dynamically created Pydantic model class with forbidden extra fields.

    Notes
    -----
    The created model will have:
    - Required fields from `schema`
    - Field descriptions from `descriptions`
    - Extra fields forbidden via ConfigDict
    """
    # 1) Turn each entry into (type, default=Field(...))
    fields: dict[str, tuple[type, Field]] = {}
    for name, dtype in schema.items():
        desc = descriptions.get(name, "")
        fields[name] = (
            dtype,
            Field(..., description=desc),
        )

    # 2) Forbid any extras via a ConfigDict
    config = ConfigDict(extra="forbid")

    # 3) Create and return the model
    Model = create_model(  # noqa: N806
        model_name,
        __config_dict__=config,
        **fields,
    )

    # 4) (Optionally) attach a docstring matching your static example
    Model.__doc__ = docstring

    return Model
