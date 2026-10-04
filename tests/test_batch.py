import nibabel as nib
import numpy as np
import pytest
from cbct_width import batch
from cbct_width.pipeline import DEFAULT_ARCH_MAPS


def case_with_mask(tmp_path, image):
    scans = tmp_path / 'scans'
    scans.mkdir()
    scan = scans / 'case001.nii.gz'
    nib.save(nib.Nifti1Image(np.zeros(image.shape, dtype=np.int16), image.affine), scan)
    mask = scans / 'case001_FULL_MASK.nii.gz'
    nib.save(image, mask)
    return batch.discover_cases(scans)[0], mask


@pytest.mark.parametrize('invalid', ['shape', 'affine', 'binary', 'fractional', 'corrupt'])
def test_mask_reuse_rejects_invalid_candidate(phantom, tmp_path, monkeypatch, invalid):
    _, image = phantom
    case, mask = case_with_mask(tmp_path, image)
    monkeypatch.setattr(batch, 'MASK_FALLBACK_ROOT', tmp_path / 'fallback')
    values = np.asanyarray(image.dataobj).copy()
    affine = image.affine.copy()
    if invalid == 'shape':
        values = values[:-1]
    elif invalid == 'affine':
        affine[0, 3] += 1
    elif invalid == 'binary':
        values = (values > 0).astype(np.uint8)
    elif invalid == 'fractional':
        values = values.astype(np.float32) + 0.5
    if invalid == 'corrupt':
        mask.write_bytes(b'not a nifti')
    else:
        nib.save(nib.Nifti1Image(values, affine), mask)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    messages = []
    assert batch._find_reusable_batch_mask(case, case['path'], scratch, messages.append) is None
    assert any('Ignoring unusable mask' in m for m in messages)


def test_reuse_and_resume_never_invoke_segmentation(phantom, tmp_path, monkeypatch):
    case, _ = case_with_mask(tmp_path, phantom[1])
    output = tmp_path / 'output'
    monkeypatch.setattr(batch, 'MASK_FALLBACK_ROOT', tmp_path / 'fallback')
    def forbidden(*args, **kwargs):
        pytest.fail('segmentation must not run for a valid existing mask')
    monkeypatch.setattr(batch, 'segment_scan', forbidden)
    kwargs = dict(arch_maps=DEFAULT_ARCH_MAPS, fold='5', device='cpu', results_dir=str(tmp_path / 'models'), log=lambda m: None)
    result, _ = batch.run_batch([case], output, **kwargs)
    assert result.iloc[0]['status'] == 'ok'
    assert bool(result.iloc[0]['segmentation_reused'])
    assert result.iloc[0]['maxilla_pred'] == pytest.approx(40, abs=0.05)
    monkeypatch.setattr(batch, 'measure_widths', forbidden)
    again, _ = batch.run_batch([case], output, **kwargs)
    assert len(again) == 1
    assert again.iloc[0]['status'] == 'ok'


def test_force_rerun_bypasses_existing_masks(phantom, tmp_path, monkeypatch):
    case, mask = case_with_mask(tmp_path, phantom[1])
    calls = []
    def fake_segment(*args, **kwargs):
        calls.append(True)
        return str(mask)
    monkeypatch.setattr(batch, 'segment_scan', fake_segment)
    results, _ = batch.run_batch([case], tmp_path / 'out', arch_maps=DEFAULT_ARCH_MAPS,
        fold='5', device='cpu', results_dir=str(tmp_path / 'models'), force_rerun=True, log=lambda m: None)
    assert calls == [True]
    assert results.iloc[0]['status'] == 'ok'
    assert not bool(results.iloc[0]['segmentation_reused'])


def test_discovery_ignores_masks_and_disambiguates_duplicate_names(tmp_path):
    for folder in ('a', 'b'):
        path = tmp_path / folder
        path.mkdir()
        (path / 'scan.nii.gz').touch()
        (path / 'scan_FULL_MASK.nii.gz').touch()
    cases = batch.discover_cases(tmp_path)
    assert len(cases) == 2
    assert len({c['case_id'] for c in cases}) == 2


def test_corrupt_resume_file_is_not_overwritten(tmp_path):
    path = tmp_path / 'batch_results.csv'
    path.write_text('"unterminated', encoding='utf-8')
    with pytest.raises(ValueError, match='Cannot read previous results'):
        batch.run_batch([], tmp_path, DEFAULT_ARCH_MAPS, '5', 'cpu', str(tmp_path / 'models'))
    assert path.read_text() == '"unterminated'
