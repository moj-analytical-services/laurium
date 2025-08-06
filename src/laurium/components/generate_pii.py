"""Generate names, IDs, etc. for synthetic data."""

from faker import Faker


def add_unique_names(input: list[str], num: int) -> list[str]:
    """Initialise input list for chain injection.

    Parameters
    ----------
    input : list[str]
        Input list passed to invoke the chain
    num : int
        User defined number of unique names


    Returns
    -------
    input_list : list[str]
        List containing original inputs appended with prompt for including a
        random name
    """
    fake = Faker()
    names = set()

    while len(names) < num:
        names.add(fake.name())

    name_list = list(names)

    input_list = [
        f"{x}. Use the following names for any people mentioned: {name_list}"
        for x in input
    ]
    return input_list
