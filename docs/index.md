# Laurium

## Turn free text data into tables for analysis

Free text data contains a wealth of information, but cannot be analysed with
traditional statistical techniques. Laurium uses LLMs to convert this free text
into tables, allowing us to unlock the information in the free text without
requiring manual data labelling.

## What does Laurium do?

For example, customer feedback stating "The login system crashed and I lost all
my work!" contains information about the sentiment of the review, how urgently
it needs to be addressed, what department is responsible for addressing the
complaints and if action is required. Laurium provides the tools to extract and
structure this information enabling quantitative analysis and data-driven
decision making:

```
                                            text sentiment  urgency department action_required
The login system crashed and I lost all my work!  negative        5         IT             yes
```

This can be scaled to datasets which would be impossible to manually review and
label.

This package started from work done by the BOLD Families project on [estimating
the number of children who have a parent in prison](
    https://www.gov.uk/government/statistics/estimates-of-children-with-a-parent-in-prison
).

## What does Laurium _not_ do?


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Why 'Laurium'

[Laurium](https://en.wikipedia.org/wiki/Lavrio) was an ancient Greek mine,
famed for its rich silver veins that fueled the rise of Athens as a
Mediterranean powerhouse.

Just as Laurium’s silver generated immense wealth for ancient Athens, so modern
text mining (based on LLMs) holds the potential to unlock huge untapped value
from unstructured information.
