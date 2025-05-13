# Converting Markdown to Jupyter Notebooks

## Setup

1. Install jupytext:
```bash
pip install jupytext
```

## Markdown Format for Notebooks

Structure your markdown files like this:
```markdown
# Title

## Section 1

Regular markdown text here.

```python
# This is a code cell
print("Hello World")
```

More markdown text.

```python
# Another code cell
import pandas as pd
```
```

## Conversion Steps

1. **Basic Conversion**
   ```bash
   jupytext --to notebook docs/notebooks/01_document_processing.md
   ```
   This creates `01_document_processing.ipynb`

2. **Paired Mode** (keeps .md and .ipynb in sync)
   ```bash
   jupytext --set-formats ipynb,md:myst docs/notebooks/01_document_processing.ipynb
   ```

3. **Convert All Notebooks**
   ```bash
   for f in docs/notebooks/*.md; do
     jupytext --to notebook "$f"
   done
   ```

## Best Practices

1. **File Organization**
   ```
   docs/
   ├── notebooks/
   │   ├── 01_document_processing.md
   │   ├── 02_vector_store.md
   │   ├── 03_rag_chain.md
   │   └── 04_semantic_search.md
   └── specs/
       └── 07-notebook-conversion.md
   ```

2. **Version Control**
   - Commit both .md and .ipynb files
   - Use jupytext's paired mode for auto-sync
   - Clear notebook outputs before committing

3. **Markdown Formatting**
   - Use ```python blocks for code cells
   - Use regular markdown for text cells
   - Add cell metadata with ```{python} #tags
   
4. **Testing Conversion**
   ```bash
   # Convert
   jupytext --to notebook test.md
   
   # Verify
   jupyter nbconvert --execute --to notebook --inplace test.ipynb
   ```

## Example Markdown Format

Here's how to format the markdown file:

````markdown
# Document Processing Examples

This notebook demonstrates document processing capabilities.

```python
# Import required libraries
from pathlib import Path
from src.document_processing import DocumentReader
```

## Setup Test Data

Create test files for examples:

```python
def create_test_files():
    # Implementation here
    pass
```

## Document Reading

Test basic document reading:

```{python} tags=["test"]
reader = DocumentReader()
result = reader.read_file('test.pdf')
print(result)
```
````

## Common Issues

1. **Code Cell Detection**
   - Ensure proper code block syntax
   - Use ```python for code cells
   - Leave blank lines around blocks

2. **Metadata Handling**
   - Use ```{python} for cell metadata
   - Add tags with tags=["example"]
   - Specify cell properties in metadata

3. **Execution**
   - New notebooks have no outputs
   - Run cells manually or use nbconvert
   - Check for missing dependencies
