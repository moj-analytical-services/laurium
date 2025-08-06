"""
Module for extracting mentions of specific keywords from text using spaCy.

spacy_tools.py defines the `SpacyPipeline` class, which utilizes spaCy to find
mentions of specified keywords within text documents.

Classes
-------
SpacyPipeline
    A pipeline class for extracting keyword mentions from text using spaCy.

Usage
-----
Import the `SpacyPipeline` class and use it to process your text data.

"""

from typing import Optional

import pandas as pd
import spacy

from laurium.components.utils import regex_convert_vertical_whitespace


class SpacyPipeline:
    """
    A pipeline class for extracting mentions of specific keywords from text.

    This class uses spaCy to find mentions of given keywords in text documents.
    It can generate context windows around the mentions and produce DataFrames
    containing the extracted information.

    Parameters
    ----------
    keywords : list of dict
        A list of keyword patterns for the spaCy matcher. Each pattern should
        be a dictionary formatted according to spaCy's matcher pattern
        specifications.
    context_window_size : int, optional
        The number of sentences to include as context around each keyword
        match. Default is 1.
    extra_punct_chars : list of str, optional
        Additional punctuation characters to consider when segmenting
        sentences. These characters are appended to spaCy's default punctuation
        characters.
    """

    def __init__(
        self,
        keywords: list[dict[str, dict[str, list[str]]]],
        context_window_size: int = 1,
        extra_punct_chars: Optional[list[str]] = None,
    ):
        self.context_window_size = context_window_size

        if extra_punct_chars is None:
            self.punct_chars = (
                spacy.pipeline.sentencizer.Sentencizer.default_punct_chars
            )
        else:
            self.punct_chars = (
                spacy.pipeline.sentencizer.Sentencizer.default_punct_chars
                + extra_punct_chars
            )

        self.spacy_model = spacy.load(
            "en_core_web_sm", exclude=["parser", "ner"]
        )
        self.spacy_model.add_pipe(
            "sentencizer", config={"punct_chars": self.punct_chars}
        )

        self.spacy_matcher = spacy.matcher.Matcher(self.spacy_model.vocab)

        self.spacy_matcher.add("keyword", [keywords])

    def get_keyword_mentions(
        self, texts: list[str]
    ) -> tuple[list[int], list[str], list[str]]:
        """
        Identify mentions of keywords and provide contexts.

        Given a list of texts, return indices of the ones containing keyword
        mentions and the contexts.

        Parameters
        ----------
        texts : list of str
            list of texts to process.

        Returns
        -------
        indices : list of int
            list of indices of texts which contain keyword mentions.
        contexts : list of str
            list of premise texts for input to NLI model.
        spans : list of str
            List of words from the context that got picked up, only first one
            per sentence.
        """
        context_size = self.context_window_size // 2

        indices = []
        contexts = []
        spans = []

        # Preprocessing: convert multiple line breaks to a single line break
        texts = [regex_convert_vertical_whitespace(x) for x in texts]

        # Set up spaCy pipeline for streaming multiprocessing
        docs = self.spacy_model.pipe(texts, n_process=8, batch_size=100)

        for i, doc in enumerate(docs):
            for j, sent in enumerate(doc.sents):
                matches = self.spacy_matcher(sent)
                if matches:
                    indices.append(i)

                    _, span_start, span_end = matches[0]
                    spans.append(sent[span_start:span_end].text.lower())

                    if context_size == 0:
                        contexts.append(sent.text.strip())
                    else:
                        contexts.append(
                            self.generate_long_context_from_doc(
                                doc, j, context_size
                            )
                        )

        return indices, contexts, spans

    def generate_long_context_from_doc(
        self, doc: spacy.tokens.Doc, sentence_index: int, context_size: int
    ) -> str:
        """
        Generate context if window size is longer than 1.

        Parameters
        ----------
        doc : spacy.tokens.Doc
            spaCy Doc object (contact note processed by spaCy model).
        sentence_index : int
            Index of the sentence with the keyword mention in the doc.
        context_size : int
            Number of additional sentences to include on each side.

        Returns
        -------
        context : str
            Concatenated sentences containing the keyword mention.
        """
        context_start = max(0, sentence_index - context_size)
        context_end = sentence_index + context_size

        context = " ".join(
            [x.text for x in list(doc.sents)[context_start : context_end + 1]]
        ).strip()

        return context

    def generate_df_context(self, df_chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Given a data chunk, returns a DataFrame containing keyword mentions.

        Parameters
        ----------
        df_chunk : pandas.DataFrame
            DataFrame containing columns 'cluster_id' and 'note_text'.

        Returns
        -------
        df_context : pandas.DataFrame
            DataFrame containing additional columns 'context' and
            'matched_word'.  One row per keyword mention.
        """
        notes = df_chunk.note_text.tolist()
        note_indices, contexts, spans = self.get_keyword_mentions(notes)

        df_context = (
            df_chunk.iloc[note_indices]
            .drop("note_text", axis=1)
            .reset_index(drop=True)
        )
        df_context["context"] = contexts
        df_context["matched_word"] = spans

        df_context = df_context.drop_duplicates(
            ["cluster_id", "context", "matched_word"]
        ).reset_index(drop=True)

        return df_context

    def generate_df_stats(
        self, df_chunk: pd.DataFrame, df_context: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Get a DataFrame containing stats for recording.

        Parameters
        ----------
        df_chunk : pandas.DataFrame
            DataFrame containing notes.
        df_context : pandas.DataFrame
            DataFrame containing keyword mentions.

        Returns
        -------
        df_stats : pandas.DataFrame
            DataFrame with columns 'cluster_id', 'n_notes', 'n_mentions'.
        """
        # Count number of notes and keyword mentions for each cluster
        df_stats = (
            df_chunk.cluster_id.value_counts()
            .rename_axis("cluster_id")
            .reset_index(name="n_notes")
        )
        n_mentions = (
            df_context.cluster_id.value_counts()
            .rename_axis("cluster_id")
            .reset_index(name="n_mentions")
        )

        df_stats = df_stats.set_index("cluster_id").join(
            n_mentions.set_index("cluster_id")
        )
        df_stats = df_stats.sort_index().fillna(0).reset_index()
        return df_stats
