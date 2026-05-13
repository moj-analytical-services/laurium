# Installation

You can install Laurium either from PyPI or from GitHub directly. If installing from PyPI, you will need to install a spaCy dependency alongside the package.

Laurium comes with two sets of features:
- **Core features** (included by default): Text extraction and analysis using
large language models
- **Advanced ML features** (optional): Fine-tuning and training encoder based
models.

### Standard Installation

For most users who want to use large language models:

#### From PyPI
```bash
# using uv (recommended)
uv add laurium https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# using pip
pip install laurium https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

#### From GitHub
```bash
# using uv
uv add git+https://github.com/moj-analytical-services/laurium.git

# using pip
pip install git+https://github.com/moj-analytical-services/laurium.git
```

### Advanced Installation

If you require encoder-only fine-tuning and training:

#### From PyPI
```bash
# using uv
uv add laurium[encoder]
uv add https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# using pip
pip install laurium[encoder]
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

#### From GitHub
```bash
# using uv
uv add "laurium[encoder] @ git+https://github.com/moj-analytical-services/laurium.git"

# using pip
pip install "laurium[encoder] @ git+https://github.com/moj-analytical-services/laurium.git"
```
