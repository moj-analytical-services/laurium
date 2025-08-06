"""A collection of utility methods used in laurium."""

import re


def regex_convert_vertical_whitespace(text: str) -> str:
    """
    Replace multiple line breaks with single LF break.

    Parameters
    ----------
    text : str
        Input text

    Returns
    -------
    str
        Text with standardised linebreaks
    """
    # Do this because some free text uses line breaks to indicate different
    # segments (instead of using punctuations such as full stop).  \n is
    # defined as a punctuation in spacy sentencizer so will be picked up, so
    # converting multiple line breaks to a single one allows these sentences to
    # be properly segmented.

    return re.sub(r"[\n\v\f\r\u2028\u2029]+", "\n", text)


def convert_tensor_to_float(tensor, precision: int = 3) -> float:
    """

    Convert a tensor to a float with a given precision.

    Parameters
    ----------
    tensor :

    precision : int
        Integer value of desired precision of output value
    """
    return round(float(tensor), precision)
