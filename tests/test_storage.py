import pandas as pd
import pytest
from cbct_width import storage


def test_atomic_csv_replaces_complete_file(tmp_path):
    path = tmp_path / 'results.csv'
    storage.atomic_write_csv(pd.DataFrame({'case': ['first'], 'width': [40.]}), path)
    storage.atomic_write_csv(pd.DataFrame({'case': ['second'], 'width': [41.]}), path)
    assert pd.read_csv(path)['case'].tolist() == ['second']
    assert list(tmp_path.iterdir()) == [path]


def test_failed_write_preserves_previous_result(tmp_path):
    path = tmp_path / 'results.csv'
    path.write_text('original results', encoding='utf-8')
    class BrokenFrame:
        def to_csv(self, stream, **kwargs):
            stream.write('partial output')
            raise OSError('simulated disk failure')
    with pytest.raises(OSError, match='simulated'):
        storage.atomic_write_csv(BrokenFrame(), path)
    assert path.read_text() == 'original results'
    assert list(tmp_path.iterdir()) == [path]


def test_failed_replace_preserves_previous_result(tmp_path, monkeypatch):
    path = tmp_path / 'results.csv'
    path.write_text('original results', encoding='utf-8')
    def fail_replace(*args):
        raise PermissionError('simulated locked destination')
    monkeypatch.setattr(storage.os, 'replace', fail_replace)
    with pytest.raises(PermissionError):
        storage.atomic_write_csv(pd.DataFrame({'case': ['new']}), path)
    assert path.read_text() == 'original results'
    assert [p for p in tmp_path.iterdir() if p != path] == []
