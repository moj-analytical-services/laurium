"""Labeller class for annotating text using an LLM."""

from typing import Any, Literal

import jinja2
from langchain_core.language_models.chat_models import BaseChatModel

from laurium.decoder_models.extractor import Extractor


class Labeller(Extractor):
    """
    AI labeller for annotating text using an LLM.

    Parameters
    ----------
    labels : dict[str, str]
        A dictionary defining the desired output from the model, where
        each key is a field name, and the value is a description of the
        label.
    llm : dict[str, Any] | BaseChatModel
        Either a dictionary of parameters to create an LLM instance or
        a pre-configured language model instance (see
        `laurium.decoder_models.llm.create_llm`).
    prompt : str, optional
        The base prompt to use for the extraction task.
        Default is "You are an expert annotator. Label the following
        text using the provided labels."
    **prompt_kwargs : dict[str, Any]
        Additional arguments to customize the prompt creation, such as
        keywords for the system message.
    """

    def __init__(
        self,
        labels: dict[str, str],
        llm: dict[str, Any] | BaseChatModel,
        prompt: str = (
            "You are an expert annotator. Label the following text using the "
            "provided labels."
        ),
        acronyms: dict[str, str] | None = None,
        **prompt_kwargs: dict[str, Any],
    ):
        # Set up schema and Pydantic model
        self.labels = labels

        schema = {
            "label": (
                Literal[tuple(labels)],  # type: ignore
                (
                    f"The label assigned to the free text; one of "
                    f"{list(labels)}."
                ),
            ),
            "explanation": (
                str,
                "The explanation for why this label was chosen.",
            ),
        }

        # Build label prompt from template
        env = jinja2.Environment(loader=jinja2.PackageLoader(__name__))
        template = env.get_template("label_prompt_template.jinja")
        system_prompt = template.render(
            base_prompt=prompt,
            labels=labels,
            acronyms=acronyms,
        )

        # Call extractor constructor
        super().__init__(
            schema=schema,
            llm=llm,
            prompt=system_prompt,
            **prompt_kwargs,
        )
