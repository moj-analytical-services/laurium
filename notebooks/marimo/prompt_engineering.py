import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # 🚀 LLM Prompt Engineering & Evaluation Notebook

    Welcome! This notebook will guide you through:

    1. **Prompt Construction** – Create and customize prompts with interactive widgets.
    2. **Schema Definition** – Dynamically define and validate output schemas.
    3. **LLM Selection** – Choose between different LLM providers and models.
    4. **Batch Extraction** – Run your prompt over an evaluation dataset.
    5. **Visualization** – Inspect performance via a confusion matrix and example table.

    ## 📋 How to Use This Notebook

    ### 🔄 Interactive Form Flow
    This notebook uses **interactive forms** that appear progressively as you complete each step:

    1. **Fill out each form completely** before clicking the **Submit** button
    2. **New sections appear** only after you submit the previous form
    3. **Forms are connected** - later forms depend on earlier choices
    4. **Submit buttons** are usually at the bottom of each form section

    ### ⚠️ Important: Making Changes
    If you need to modify any information after submitting:

    - **You must re-submit ALL subsequent forms** in order
    - The notebook flows top-to-bottom like a pipeline
    - Changing an early form (like LLM provider) resets everything below it
    - Think of it like dominoes - changing one affects all the ones after it

    ### 🎯 Step-by-Step Process
    1. **Start** → Select LLM Provider → **Submit**
    2. **Wait** → New form appears with model options
    3. **Configure** → Fill all fields in the new form → **Submit**
    4. **Repeat** → Continue until you reach the final visualization

    ### 💡 Pro Tips
    - **Read instructions carefully** before filling forms - some fields must match exactly
    - **Check for stop messages** - they indicate what needs to be submitted
    - **Be patient** - some steps (like batch extraction) take time to process
    - **Don't skip ahead** - each section builds on the previous one

    Ready? Let's start with selecting your LLM provider below! 👇
    """
    )
    return


@app.cell
def _(mo):
    with mo.status.spinner(
        title="Loading dataset from huggingface..."
    ) as _spinner:
        # Prepare data and splits
        from datasets import load_dataset

        classifier_tomatoes = load_dataset("rotten_tomatoes")
        labelled_dataset = (
            classifier_tomatoes["train"]
            .to_pandas()
            .sample(100)
            .reset_index(drop=True)
        )
        _spinner.update(subtitle=f"Done, loaded {len(labelled_dataset)} rows")
    return (labelled_dataset,)


@app.cell
def _(mo):
    keywords_ui = mo.ui.text(value="good,boring, terrible")
    final_query_ui = mo.ui.text(value="analyse this text {text}")
    llm_provider_ui = mo.ui.dropdown(
        options=["bedrock", "ollama"], label="LLM Provider"
    )

    # Default pre-populated model
    model_defaults = {
        "bedrock": "anthropic.claude-3-haiku-20240307-v1:0",
        "ollama": "qwen2.5:7b",
    }

    provider_regions = {
        "bedrock": ["eu-west-1", "eu-west-2"],
    }

    model_family_options = {"bedrock": ["anthropic"]}
    return (
        final_query_ui,
        keywords_ui,
        llm_provider_ui,
        model_defaults,
        model_family_options,
        provider_regions,
    )


@app.cell
def _(llm_provider_ui, mo):
    llm_provider_md = (
        mo.md("""
    ## 🤖 LLM Configuration

    Welcome to the LLM setup! Here's how to get started:

    **🎯 Quick Start:** Select your preferred LLM provider from the dropdown below, then click **Submit** to unlock model-specific options.

    <p class='subsection'>
    <strong>Provider Options:</strong>
    </p>
    <ul class='subsection'>
      <li>☁️ <strong>Bedrock</strong>: AWS-hosted models (Claude family) - requires AWS credentials</li>
      <li>🖥️ <strong>Ollama</strong>: Local models running on your machine - requires Ollama installed</li>
    </ul>

    {provider}

    <p class='subsection'>
    💡 <strong>Tip:</strong> Choose Bedrock for production-grade performance, or Ollama for local experimentation without cloud costs.
    </p>
        """)
        .batch(provider=llm_provider_ui)
        .form()
    )
    llm_provider_md
    return (llm_provider_md,)


@app.cell
def _(
    final_query_ui,
    keywords_ui,
    llm_provider_md,
    mo,
    model_defaults,
    model_family_options,
    provider_regions,
):
    mo.stop(
        llm_provider_md.value is None,
        mo.md(
            "**⚠️ Action Required:** Please select a provider and click **Submit** in the LLM Configuration section above to continue setting up your model."
        ),
    )
    llm_ui = mo.ui.text(
        value=""
        if llm_provider_md.value is None
        else model_defaults.get(llm_provider_md.value["provider"], ""),
        label=f"{llm_provider_md.value['provider'].title()} Models"
        if llm_provider_md.value["provider"]
        else "Models",
    )

    llm_model_family_ui = (
        mo.ui.dropdown(
            options=model_family_options.get(
                llm_provider_md.value["provider"], [None]
            ),
            label=f"{llm_provider_md.value['provider'].title()} model family",
        )
        if llm_provider_md.value is not None
        and llm_provider_md.value["provider"] == "bedrock"
        else None
    )

    region_name_ui = mo.ui.dropdown(
        options=provider_regions.get(
            llm_provider_md.value["provider"], [None]
        ),
        label=f"{llm_provider_md.value['provider'].title()} Regions",
    )

    temperature_ui = mo.ui.slider(start=0.0, stop=1.0, step=0.1, value=0.0)

    text_column_ui = mo.ui.text(
        placeholder="Provide column name for processing",
        value="text",
        label="Name of column containing text:",
    )

    # Build the markdown string conditionally
    is_bedrock = (
        llm_provider_md.value is not None
        and llm_provider_md.value["provider"] == "bedrock"
    )

    # Build configuration section
    config_items = []

    if is_bedrock:
        config_items.append("""  <li>⚙️ <strong>Model family</strong>: The AI company/architecture (e.g., Anthropic for Claude models)
      <br><em>💡 Different families have different strengths - Anthropic excels at analysis tasks</em></li>
      {aws_model_family}""")

    config_items.append("""  <li>🤖 <strong>Model name</strong>: The specific model version
      <br><em>💡 Larger models are more capable but slower/costlier</em></li>
      {llm}""")

    config_items.append("""  <li>🌡️ <strong>Temperature</strong>: Controls creativity vs consistency
      <br><em>💡 Use 0.0 for reproducible analysis, 0.5-1.0 for creative tasks</em></li>
      {temperature}""")

    if is_bedrock:
        config_items.append("""  <li>📍 <strong>Region</strong>: Where your LLM runs
      <br><em>💡 Choose closest region for speed, or specific regions for compliance</em></li>
      {region_name}""")

    config_section = "\n".join(config_items)

    # Build the complete markdown - NO f-string!
    markdown_content = (
        """## 🎛️ Model Configuration Details
    Now let's fine-tune your LLM settings for optimal results:
    <p class='subsection'>
    <strong>📋 Step-by-Step Guide:</strong>
    </p>
    <ul class='subsection'>
    """
        + config_section
        + """
    </ul>
    ## 📝 Prompt Engineering Workshop
    Time to craft your perfect prompt! This is where the magic happens:
    <p class='subsection'>
    <strong>🎨 Prompt Components:</strong>
    </p>
    <ul class='subsection'>
      <li>✏️ <strong>Base prompt</strong>: Your main instructions to the LLM
      <br><em>💡 Include context, examples, and clear output format expectations</em>
      <br><em>🔴 <strong>CRITICAL:</strong> Your JSON examples here MUST use the exact field names you'll define in the schema later!</em>
      <br><em>📌 Example: If you use {{"ai_label": 1}} in your prompt, you MUST name your field "ai_label" in the schema</em></li>
      {base_prompt}
      <li>🔑 <strong>Keywords</strong>: Important terms to emphasize (comma-separated)
      <br><em>💡 Example: "positive,negative,neutral" for sentiment analysis</em></li>
      {keywords}
      <li>🧩 <strong>Final query</strong>: The template for each data point
      <br><em>💡 Use {{text}} as placeholder - it will be replaced with actual data</em>
      <br><em>📌 Example: "Analyze this review: {{text}}"</em></li>
      {final_query}
    </ul>
    <p class='subsection'>
    ⚠️ <strong>Remember your field names!</strong> Whatever JSON keys you use in your prompt examples (like "ai_label", "sentiment", etc.) must exactly match the field names you'll define in the schema step.
    </p>
    ## 🗂️ Dataset Configuration
    Almost there! Tell us which column contains your text data:
    <p class='subsection'>
    <strong>📊 Data Setup:</strong>
    </p>
    <ul class='subsection'>
      <li>📝 <strong>Column Name</strong>: The exact name of your text column
      <br><em>💡 Default: "text" (works with the sample dataset)</em>
      <br><em>⚠️ Only change this if you're using a different dataset with a different column name</em>
      <br><em>🔍 Case-sensitive! "Text" ≠ "text"</em></li>
      {text_column}
    </ul>
    <p class='subsection'>
    🚀 <strong>Ready?</strong> Click **Submit** to configure your extraction pipeline!
    </p>
    """
    )

    markdown = mo.md(markdown_content)

    prompts_ui = mo.ui.text_area(
        value="""
        You are a sentiment analysis assistant for Rotten Tomatoes reviews.
        Given the review below, determine whether its sentiment is positive or negative.
        **Instructions:**
        - Output **ONLY** valid JSON with exactly one key: "ai_label".
        - Use 1 for positive sentiment.
        - Use 0 for negative sentiment.
        - Do not include any extra text.
        Review:
        "I absolutely loved the performances and story. A must-watch!"
        Your response:
        {{"ai_label":1}}
        ---
        Review:
        "The plot was dull and the acting was uninspired. I walked out early."
        Your response:
        {{"ai_label":0}}
    """
    )

    # Build the batch UI dictionary
    batch_ui_dict = {
        "base_prompt": prompts_ui,
        "keywords": keywords_ui,
        "final_query": final_query_ui,
        "llm": llm_ui,
        "temperature": temperature_ui,
        "text_column": text_column_ui,
    }

    # Only add bedrock-specific UI elements if bedrock is selected
    if is_bedrock:
        batch_ui_dict["aws_model_family"] = llm_model_family_ui
        batch_ui_dict["region_name"] = region_name_ui

    batch = mo.ui.batch(
        markdown,
        batch_ui_dict,
    ).form()
    batch
    return (batch,)


@app.cell
def _(batch, llm, llm_provider_md, mo, prompts):
    mo.stop(
        batch.value is None,
        mo.md(
            "**⚠️ Action Required:** Please fill out LLM and Prompt configuration form and click **Submit**."
        ),
    )
    ### building system prompt
    with mo.status.spinner(
        title="Building prompt and configuring LLM"
    ) as _spinner:
        _spinner.update(subtitle="Building prompt")
        system_message = prompts.create_system_message(
            base_message=batch.value.get("base_prompt"),
            keywords=batch.value.get("keywords").split(","),
        )

        ### building full prompt
        extraction_prompt = prompts.create_prompt(
            system_message=system_message,
            examples=None,
            example_human_template=None,
            example_assistant_template=None,
            final_query=batch.value.get("final_query"),
        )

        ### llm
        _spinner.update(subtitle="Configuring LLM")
        llm_instance = llm.create_llm(
            llm_platform=llm_provider_md.value.get("provider"),
            model_name=batch.value.get("llm"),
            temperature=batch.value.get("temperature"),
            aws_region_name=batch.value.get("region_name"),
        )

        _spinner.update(subtitle="Complete")
    return extraction_prompt, llm_instance


@app.cell
def _():
    ### Cell contains code to save prompt messages locally - uncomment to save prompts locally

    # import os
    # from datetime import datetime

    # save_dir = "prompts"
    # os.makedirs(save_dir, exist_ok=True)

    # timestamp = datetime.now()
    # filename = f"prompt_{timestamp}.txt"
    # filepath = os.path.join(save_dir, filename)

    # with open(filepath, "w", encoding="utf-8") as f:
    #    f.write(system_message)
    return


@app.cell
def _(batch, mo):
    mo.stop(
        batch.value is None,
        mo.md(
            "**⚠️ Action Required:** Please fill out LLM and Prompt configuration form and click **Submit**."
        ),
    )
    ### schema definition logic
    num_fields_ui = mo.ui.number(
        start=1,
        stop=10,
        step=1,
        value=1,
        label="How many fields will the LLM output?",
    )
    parser_markdown = mo.md("""
    ## 🛡️ Output Schema Builder

    Let's define exactly what you want the LLM to extract:

    <p class='subsection'>
    <strong>🏗️ Schema Design:</strong>
    </p>
    <ul class='subsection'>
      <li>🔢 <strong>Number of fields</strong>: How many different pieces of information to extract
      <br><em>💡 Count the unique keys in your prompt's JSON examples</em>
      <br><em>📌 Example: {{"ai_label": 1}} = 1 field, {{"sentiment": "positive", "confidence": 0.9}} = 2 fields</em>
      <br><em>🔴 <strong>Must match your prompt!</strong> If your examples show 1 JSON key, select 1 field</em></li>
    </ul>

    {num_fields}

    <p class='subsection'>
    ⚡ <strong>Pro tip:</strong> Look at your prompt examples and count how many different JSON keys you used. That's your number!
    </p>
    """)
    parser_markdown_batch = mo.ui.batch(
        parser_markdown, {"num_fields": num_fields_ui}
    ).form()
    parser_markdown_batch
    return (parser_markdown_batch,)


@app.cell
def _(mo, parser_markdown_batch):
    mo.stop(
        parser_markdown_batch.value is None,
        mo.md(
            "**⚠️ Action Required:** Please fill out Output Schema Builder configuration form and click **Submit**."
        ),
    )
    define_schema_markdown = mo.md(
        """
    ## 🛠️ Define Your Output Fields

    Time to specify each field in detail. This ensures the LLM outputs exactly what you need:

    <p class='subsection'>
    🔴 <strong>CRITICAL ALIGNMENT CHECK:</strong>
    <br>Look back at your prompt examples. What JSON keys did you use? You MUST use those exact same field names here!
    </p>

    <p class='subsection'>
    <strong>📐 Field Configuration Guide:</strong>
    </p>
    <ul class='subsection'>
      <li>🏷️ <strong>Field name</strong>: The JSON key for this data point
      <br><em>🔴 <strong>MUST MATCH YOUR PROMPT!</strong> If your prompt shows {{"ai_label": 1}}, use "ai_label" here</em>
      <br><em>💡 Common names: "ai_label", "sentiment", "category", "score"</em>
      <br><em>⚠️ Case-sensitive and must be exact!</em></li>

      <li>📊 <strong>Field type</strong>: The data type to enforce
      <br><em>💡 Match the type to your prompt examples:</em>
      <br>&nbsp;&nbsp;• Used {{"ai_label": 1}} in prompt? → Choose <code>int</code>
      <br>&nbsp;&nbsp;• Used {{"sentiment": "positive"}} in prompt? → Choose <code>str</code>
      <br>&nbsp;&nbsp;• Used {{"confidence": 0.95}} in prompt? → Choose <code>float</code>
      <br>&nbsp;&nbsp;• Used {{"is_positive": true}} in prompt? → Choose <code>bool</code></li>

      <li>💬 <strong>Field description</strong>: What this field represents
      <br><em>💡 This documents a description of the field in the pydantic object and is not interpreted by the LLM </em>
    </ul>

    <p class='subsection'>
    🎯 <strong>Example - If your prompt contains:</strong>
    <pre>
    Your response:
    {{"ai_label": 1}}
    </pre>
    <strong>Then configure:</strong>
    <br>• Name: <code>ai_label</code> (exact match!)
    <br>• Type: <code>int</code> (because 1 is an integer)
    <br>• Description: "Sentiment classification where 1=positive, 0=negative"
    </p>

    {field_defs}

    <p class='subsection'>
    ✅ <strong>Ready?</strong> Click **Submit** to generate your Pydantic model and start extraction!
    </p>
    """
    )
    return (define_schema_markdown,)


@app.cell
def _(mo, parser_markdown_batch):
    mo.stop(
        parser_markdown_batch is None,
        mo.md("Submit to continue defining fields"),
    )
    return


@app.cell
def _(define_schema_markdown, mo, parser_markdown_batch):
    field_defs = mo.ui.array(
        [
            mo.ui.dictionary(
                {
                    "name": mo.ui.text(label="Field name"),
                    "type": mo.ui.dropdown(
                        options=[
                            "str",
                            "int",
                            "float",
                            "bool",
                            "date",
                            "datetime",
                        ],
                        label="Field type",
                    ),
                    "description": mo.ui.text_area(label="Field description"),
                }
            )
            for _ in range(int(parser_markdown_batch.value["num_fields"]))
        ]
    )

    define_schema_batch = mo.ui.batch(
        define_schema_markdown, {"field_defs": field_defs}
    ).form()
    return (define_schema_batch,)


@app.cell
def _(define_schema_batch):
    define_schema_batch
    return


@app.cell
def _(
    PydanticOutputParser,
    batch,
    define_schema_batch,
    extract,
    extraction_prompt,
    labelled_dataset,
    llm_instance,
    mo,
    pydantic_models,
):
    mo.stop(
        define_schema_batch.value is None,
        mo.md("Submit to build your pydantic model"),
    )
    values = define_schema_batch.value.get("field_defs")
    schema = {
        name: eval(typ)
        for name, typ in [(f["name"], f["type"]) for f in values]
    }
    field_descriptions = {f["name"]: f["description"] for f in values}

    CustomModel = pydantic_models.make_dynamic_example_model(
        schema, field_descriptions, model_name="MyDynamicModel"
    )

    with mo.status.spinner(title="Running batch extractor:") as _spinner:
        ### extractor
        extractor = extract.BatchExtractor(
            llm=llm_instance,
            prompt=extraction_prompt,
            parser=PydanticOutputParser(pydantic_object=CustomModel),
        )

        # Process the chunk using BatchExtractor
        processed_df = extractor.process_chunk(
            chunk_df=labelled_dataset,
            text_column=batch.value.get("text_column"),
        )

        _spinner.update(subtitle="Batch extractor complete")
    return CustomModel, processed_df


@app.cell
def _(CustomModel, processed_df):
    # Get the expected output fields from your schema
    expected_fields = list(CustomModel.model_fields.keys())

    # Check if the processing actually worked
    failure_detected = False
    error_message = ""

    if processed_df is None:
        failure_detected = True
        error_message = "Batch processing returned no results"
    else:
        # Check if all values for expected fields are None/empty
        for field in expected_fields:
            if field in processed_df.columns:
                non_null_count = processed_df[field].notna().sum()
                if non_null_count == 0:
                    failure_detected = True
                    error_message = f"All values for '{field}' are empty - likely parsing failure"
                    break
    return error_message, failure_detected


@app.cell
def _(
    CustomModel,
    define_schema_batch,
    error_message,
    failure_detected,
    mo,
    processed_df,
):
    mo.stop(
        failure_detected is True,
        mo.md(f"""
        ## ❌ Batch Processing Failed

        **Error:** {error_message}

        **Your Schema Definition:**
        ```
        {
            "\n".join(
                [
                    f"- {name}: {field.annotation}"
                    for name, field in CustomModel.model_fields.items()
                ]
            )
        }
        ```

        **Common Issues:**

        1. **Type Mismatch**: Your prompt outputs don't match the schema types
           - Example: Outputting `"ai_label": 0` when schema expects `ai_label: str`
           - Fix: Either change prompt to output `"ai_label": "0"` or change schema to `ai_label: int`

        2. **Missing Fields**: Your prompt doesn't output all required fields
           - Check that your prompt outputs ALL fields defined in your schema

        3. **Invalid JSON**: Your prompt might be outputting malformed JSON
           - Ensure your prompt examples show valid JSON format

        **Next Steps:**
        1. Review your prompt examples and ensure they match your schema exactly
        2. Test with a single example first before batch processing
        3. Check that field names in prompt match schema field names (case-sensitive!)
        """).callout(kind="danger"),
    )

    import altair as alt

    chart = mo.ui.altair_chart(
        alt.Chart(processed_df)
        .mark_rect()
        .encode(
            # horizontal: predicted label
            x=alt.X(
                f"{define_schema_batch.value.get('field_defs')[0]['name']}:N",
                title="Predicted",
            ),
            # vertical: actual label
            y=alt.Y("label:N", title="Actual"),
            # cell color = how many rows fall into each (predicted, actual) bin
            color=alt.Color(
                "count():Q", scale=alt.Scale(scheme="greenblue"), title="Count"
            ),
        )
        .properties(width=500, height=500)
        + alt.Chart(processed_df)
        .mark_text(color="black")
        .encode(
            x=alt.X(
                f"{define_schema_batch.value.get('field_defs')[0]['name']}:N"
            ),
            y=alt.Y("label:N"),
            # put the raw count in the center of each cell
            text=alt.Text("count():Q"),
        )
        .properties(width=500, height=500)
    )

    evaluation_vis_markdown = mo.md("""
    ## 🔍 Performance Analysis Dashboard

    Great job! Your model has processed the dataset. Let's analyze the results:

    <p class='subsection'>
    <strong>📊 Understanding Your Confusion Matrix:</strong>
    </p>
    <ul class='subsection'>
      <li>🎯 <strong>Perfect predictions</strong>: Look for high counts on the diagonal (top-left to bottom-right)</li>
      <li>❌ <strong>Misclassifications</strong>: Off-diagonal cells show where the model got confused</li>
      <li>🎨 <strong>Color intensity</strong>: Darker = more examples in that category</li>
    </ul>

    <p class='subsection'>
    <strong>🖱️ Interactive Features:</strong>
    </p>
    <ul class='subsection'>
      <li>👆 <strong>Click any cell</strong> in the heatmap to filter examples</li>
      <li>📋 <strong>Review specific cases</strong> in the table below to understand errors</li>
      <li>🔍 <strong>Pattern spotting</strong>: Look for systematic biases or confusion patterns</li>
    </ul>

    <p class='subsection'>
    💡 <strong>Improvement tips:</strong>
    <br>• High off-diagonal counts? → Refine your prompt with clearer examples
    <br>• Random errors? → Try a larger model or adjust temperature
    <br>• Specific confusion? → Add targeted examples to your prompt
    </p>
    """)
    return chart, evaluation_vis_markdown


@app.cell
def _(
    chart,
    define_schema_batch,
    evaluation_vis_markdown,
    failure_detected,
    mo,
):
    mo.stop(
        failure_detected is False,
        mo.vstack(
            [
                evaluation_vis_markdown,
                chart.center(),
                mo.ui.table(
                    data=chart.value[
                        [
                            "text",
                            "label",
                            define_schema_batch.value.get("field_defs")[0][
                                "name"
                            ],
                        ]
                    ],
                    show_column_summaries=False,
                    wrapped_columns=[
                        "text",
                        "label",
                        define_schema_batch.value.get("field_defs")[0]["name"],
                    ],
                ),
            ]
        ),
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from langchain_core.output_parsers import PydanticOutputParser

    return (PydanticOutputParser,)


@app.cell
def _():
    from laurium.decoder_models import extract, llm, prompts, pydantic_models

    return extract, llm, prompts, pydantic_models


if __name__ == "__main__":
    app.run()
