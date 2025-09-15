# Laurium

A Python package for extracting structured data from text and generating
synthetic data using language models.

Organisations collect vast amounts of free text data containing untapped
information that could provide decision makers with valuable insights.
Laurium addresses this by providing tools for converting unstructured text into
structured data using Large Language Models.Through prompt engineering, the
package can be adapted to different use cases and data extraction requirements,
unlocking the value hidden in text data.

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

## Install Laurium

You can install Laurium either from PyPI or from GitHub directly. If installing
from PyPI, you will need to install a spaCy dependency alongside the package.

### From GitHub

```bash
# using uv
uv add git+https://github.com/moj-analytical-services/laurium.git

# using pip
pip install git+https://github.com/moj-analytical-services/laurium.git
```

### From PyPI

```bash
# using uv
uv add laurium https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# using pip
pip install laurium https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

## LLM Provider Setup
Laurium works with both local and cloud-based language models:

### Local Models with Ollama
For running models locally without API costs:

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Spin up a local ollama server by running in your terminal: `ollama serve`
3. Pull a model by running in your terminal: `ollama pull qwen2.5:7b`

**Benefits:**

- No API costs or rate limits

- Data stays local for privacy

- Works offline

**Requirements:**

- Sufficient disk space for model storage

- GPU recommended for faster processing

### AWS Bedrock Models
For cloud-based models like Claude:

1. **AWS Account** with Bedrock service enabled

2. **Configure AWS credentials** via AWS CLI, environment variables, or IAM
   roles

3. **Bedrock permissions** for your AWS user/role

## Basic Usage

### Text Classification Pipeline

Laurium specializes in structured data extraction from text. Here's how to
build a classification pipeline:

> [!TIP]
> **Automatic Schema Integration**:
>
> As of the latest version, you can now pass `schema` and `descriptions`
> directly to `create_prompt()`. This automatically formats the JSON structure
> and field descriptions in your prompt, eliminating the need to manually
> specify the exact JSON format and ensuring consistency between your prompt
> and Pydantic schema.


#### Using Ollama (Local)
```python
from laurium.decoder_models import llm, prompts, pydantic_models, extract
from langchain_core.output_parsers import PydanticOutputParser
import pandas as pd

# 1. Create LLM instance
sentiment_llm = llm.create_llm(
    llm_platform="ollama", model_name="qwen2.5:7b", temperature=0.0
)

# 2. Define output schema once
schema = {"ai_label": int}  # 1 for positive, 0 for negative
descriptions = {
    "ai_label": "Sentiment classification (1=positive, 0=negative)"
}

# 3. Build prompt with automatic schema integration
system_message = prompts.create_system_message(
    base_message="You are a sentiment analysis assistant. Use 1 for positive"/
    "sentiment, 0 for negative sentiment.",
    keywords=["positive", "negative"],
)

extraction_prompt = prompts.create_prompt(
    system_message=system_message,
    examples=None,
    example_human_template=None,
    example_assistant_template=None,
    final_query="Analyze this text: {text}",
    schema=schema,  # Automatically formats JSON structure in prompt
    descriptions=descriptions,  # Provides field context to LLM
)

# 4. Create Pydantic model using same schema
OutputModel = pydantic_models.make_dynamic_example_model(
    schema=schema, descriptions=descriptions, model_name="SentimentOutput"
)

# 5. Create extractor and process data
parser = PydanticOutputParser(pydantic_object=OutputModel)
extractor = extract.BatchExtractor(
    llm=sentiment_llm, prompt=extraction_prompt, parser=parser
)

# Process your data
data = pd.DataFrame(
    {
        "text": [
            "I absolutely love this product!",
            "This is terrible, worst purchase ever.",
            "Great value for money, highly recommend!",
        ]
    }
)

results = extractor.process_chunk(data, text_column="text")
print(results.to_string(index=False))
```

#### Using AWS Bedrock
```python
# Same code as above, but create LLM with Bedrock:
sentiment_llm = llm.create_llm(
    llm_platform="bedrock",
    model_name="claude-3-sonnet",
    temperature=0.0,
    aws_region_name="eu-west-1",
)
# ... rest of the code remains the same
```

This will output something like:
```
                                    text  ai_label
         I absolutely love this product!         1
  This is terrible, worst purchase ever.         0
Great value for money, highly recommend!         1
```

### Multi-Field Extraction

#### Define Complex Output Schemas
Extract multiple pieces of structured data simultaneously:

```python
# Create LLM instance
feedback_llm = llm.create_llm(
    llm_platform="ollama", model_name="qwen2.5:7b", temperature=0.0
)

# Schema for analyzing customer feedback using Literal types for constraints
from typing import Literal

schema = {
    "sentiment": Literal["positive", "negative", "neutral"],
    "urgency": int,  # 1-5 scale
    "department": Literal["IT", "Support", "Product", "Sales", "Other"],
    "action_required": Literal["yes", "no"],
}

descriptions = {
    "sentiment": "Customer's emotional tone",
    "urgency": "How quickly this needs attention (1=low, 5=urgent)",
    "department": "Which department should handle this",
    "action_required": "Whether immediate action is needed",
}

# Build prompt with automatic schema integration
system_message = prompts.create_system_message(
    base_message="Analyze customer feedback and extract structured information.",
    keywords=["urgent", "complaint", "praise", "bug", "feature"],
)

# Schema automatically added to prompt - no manual JSON formatting needed!
extraction_prompt = prompts.create_prompt(
    system_message=system_message,
    examples=None,  # We'll add examples in the next section
    example_human_template=None,
    example_assistant_template=None,
    final_query="Feedback: {text}",
    schema=schema,  # Automatically shows allowed values and types
    descriptions=descriptions,  # Provides field context to LLM
)

FeedbackModel = pydantic_models.make_dynamic_example_model(
    schema=schema,
    descriptions=descriptions,
    model_name="CustomerFeedbackAnalysis",
)
```

#### Improve Accuracy with Examples
Add few-shot examples to guide the model:

```python
# Training examples for better extraction - JSON format must match schema
few_shot_examples = [
    {
        "text": "System is down, can't access anything!",
        "sentiment": "negative",
        "urgency": 5,
        "department": "IT",
        "action_required": "yes",
    },
    {
        "text": "Love the new interface design",
        "sentiment": "positive",
        "urgency": 1,
        "department": "Product",
        "action_required": "yes",
    },
]

extraction_prompt = prompts.create_prompt(
    system_message=system_message,
    examples=few_shot_examples,
    example_human_template="Feedback: {text}",
    example_assistant_template="""{{
        "sentiment": "{sentiment}",
        "urgency": {urgency},
        "department": "{department}",
        "action_required": "{action_required}"
    }}""",
    final_query="Feedback: {text}",
    schema=schema,  # Schema formatting still included with examples
    descriptions=descriptions,
)

# Create extractor and process sample data
parser = PydanticOutputParser(pydantic_object=FeedbackModel)
extractor = extract.BatchExtractor(
    llm=feedback_llm,  # your LLM instance
    prompt=extraction_prompt,
    parser=parser,
)

# Sample customer feedback data
feedback_data = pd.DataFrame(
    {
        "text": [
            "The login system crashed and I lost all my work!",
            "Really appreciate the new dark mode feature",
            "Can we get a mobile app version soon?",
            "Billing charged me twice this month, need help",
        ]
    }
)

results = extractor.process_chunk(feedback_data, text_column="text")
print(results.to_string(index=False))
```

This will output something like:
```
                                            text sentiment  urgency department action_required
The login system crashed and I lost all my work!  negative        5         IT             yes
     Really appreciate the new dark mode feature  positive        2    Product              no
           Can we get a mobile app version soon?   neutral        3    Product             yes
  Billing charged me twice this month, need help  negative        3    Support             yes
```

## Supported Models

### Ollama (Local)
Use any model available in Ollama.

### AWS Bedrock (Cloud)
Supported Bedrock models:
- `claude-3-sonnet` - Best for complex extraction tasks
- `claude-3-haiku` - Faster, cost-effective option

## Modules Reference

| Module | Sub-module | Description |
|--------|------------|-------------|
| **decoder_models** | `llm` | Create and manage LLM instances from Ollama and AWS Bedrock |
| | `prompts` | Create and manage prompt templates with optional few-shot examples |
| | `extract` | Efficient batch processing of text using LLMs |
| | `pydantic_models` | Dynamic Pydantic models for structured LLM output |
| **components** | `extract_context` | Extract keyword mentions with configurable context windows |
| | `evaluate` | Compute evaluation metrics for model predictions |
| | `load` | Load and chunk data from various sources (CSV, SQL, etc.) |
| **encoder_models** | `nli` | Natural Language Inference models for text analysis |
| | `fine_tune` | Fine-tune transformer models for custom tasks |

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Why 'Laurium'

[Laurium](https://en.wikipedia.org/wiki/Lavrio) was an ancient Greek mine,
famed for its rich silver veins that fueled the rise of Athens as a
Mediterranean powerhouse.

Just as Laurium’s silver generated immense wealth for ancient Athens, so modern
text mining (based on LLMs) holds the potential to unlock huge untapped value
from unstructured information.

## Contact Us
Please reach out to the AI for Linked Data team at AI_for_linked_data@justice.gov.uk
or bold@justice.gov.uk.
