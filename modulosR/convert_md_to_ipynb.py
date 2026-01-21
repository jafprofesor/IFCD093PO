import re
from nbformat import v4 as nbf
import json

def convert_md_to_ipynb(md_file, ipynb_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()

    lines = content.split('\n')
    cells = []
    current_markdown = []
    in_code = False
    current_code = []

    for line in lines:
        if line.strip() == '```python':
            if current_markdown:
                cells.append(nbf.new_markdown_cell('\n'.join(current_markdown)))
                current_markdown = []
            in_code = True
            current_code = []
        elif line.strip() == '```' and in_code:
            cells.append(nbf.new_code_cell('\n'.join(current_code)))
            in_code = False
            current_code = []
        elif in_code:
            current_code.append(line)
        else:
            current_markdown.append(line)

    # Add remaining markdown
    if current_markdown:
        cells.append(nbf.new_markdown_cell('\n'.join(current_markdown)))

    # Create notebook
    nb = nbf.new_notebook()
    nb.cells = cells

    # Write to file
    with open(ipynb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

# Run it
convert_md_to_ipynb('06RE_arboles_desicion_svm.md', '06RE_arboles_desicion_svm.ipynb')
