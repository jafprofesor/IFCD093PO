import json
import os

nb_path = "13_extra_eda_avanzado_limpieza.ipynb"

if os.path.exists(nb_path):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    updated = False
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            new_source = []
            for line in cell['source']:
                if "org_embarked" in line:
                    new_line = line.replace("org_embarked", "embarked")
                    new_source.append(new_line)
                    updated = True
                else:
                    new_source.append(line)
            cell['source'] = new_source
            
    if updated:
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Fixed 'org_embarked' typo successfully.")
    else:
        print("Could not find 'org_embarked' in any code cell.")
else:
    print(f"File {nb_path} not found.")
