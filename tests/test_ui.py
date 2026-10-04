from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_streamlit_starts_without_models_or_data(tmp_path, monkeypatch):
    monkeypatch.setenv('CBCT_WORK_DIR', str(tmp_path))
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / 'app.py'), default_timeout=30).run()
    assert not app.exception
    assert app.title[0].value == '🦷 CBCT Transverse Basal-Bone Width'
