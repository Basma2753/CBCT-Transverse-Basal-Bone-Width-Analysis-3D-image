import json
import os
import subprocess
import sys
from cbct_width.cli import main


def test_measure_cli_writes_strict_json(phantom, tmp_path):
    out = tmp_path / 'result.json'
    assert main(['measure', str(phantom[0]), '--output', str(out)]) == 0
    data = json.loads(out.read_text())
    assert data['arch_widths_mm']['maxilla'] > 0
    assert 'NaN' not in out.read_text()


def test_missing_scan_is_nonzero(tmp_path, capsys):
    assert main(['measure', str(tmp_path / 'missing.nii.gz')]) == 2
    assert 'not found' in capsys.readouterr().err


def test_importing_engine_does_not_import_streamlit_or_torch():
    subprocess.run([sys.executable, '-c',
        "import sys; import cbct_width.batch; assert 'streamlit' not in sys.modules; assert 'torch' not in sys.modules"], check=True)


def test_installed_cli_works_outside_checkout(tmp_path):
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-m', 'cbct_width', '--help'], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'measure' in result.stdout


def test_demo_creates_measured_phantom(tmp_path):
    assert main(['demo', '--output', str(tmp_path)]) == 0
    assert (tmp_path / 'synthetic_FULL_MASK.nii.gz').is_file()
    result = json.loads((tmp_path / 'measurements.json').read_text())
    assert abs(result['arch_widths_mm']['maxilla'] - 40) < 0.05


def test_batch_rejects_output_inside_scan_tree(tmp_path, capsys):
    assert main(['batch', str(tmp_path), '--output', str(tmp_path / 'results')]) == 2
    assert 'outside the scan directory' in capsys.readouterr().err
