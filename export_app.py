"""Export the notebook's application cell without executing the notebook."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Fail if app.py differs from the notebook; do not write files.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    notebook = json.loads((root / 'CBCT_Width_Analysis.ipynb').read_text(encoding='utf-8'))
    cells = []
    for cell in notebook['cells']:
        source = cell.get('source', [])
        source = source if isinstance(source, str) else ''.join(source)
        if cell['cell_type'] == 'code' and source.splitlines()[:1] == ['%%writefile app.py']:
            cells.append(source.split('\n', 1)[1])
    if len(cells) != 1:
        parser.error(f'Expected exactly one %%writefile app.py cell; found {len(cells)}.')
    source = cells[0]
    compile(source, 'app.py', 'exec')
    destination = root / 'app.py'
    if args.check:
        if not destination.exists() or destination.read_text(encoding='utf-8') != source:
            print('app.py differs from the notebook. Run: python export_app.py')
            return 1
        print('app.py matches the notebook; Python syntax is valid.')
        return 0
    destination.write_text(source, encoding='utf-8', newline='\n')
    print(f'Exported {destination.name} from CBCT_Width_Analysis.ipynb.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
