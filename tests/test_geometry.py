import nibabel as nib
import numpy as np
import pytest
from cbct_width.geometry import _apply_affine, _affine_for_crop, _principal_axes
from cbct_width.pipeline import measure_widths


def test_anisotropic_affine_and_crop_preserve_patient_coordinates():
    affine = np.array([[0, -0.8, 0, 17], [0.4, 0, 0, -9], [0, 0, 1.5, 4], [0, 0, 0, 1.]])
    bounds = (slice(7, 20), slice(11, 25), slice(3, 14))
    crop = _affine_for_crop(affine, bounds)
    local = np.array([[0, 0, 0], [2, 3, 4]])
    expected = nib.affines.apply_affine(affine, local + [7, 11, 3])
    np.testing.assert_allclose(_apply_affine(local, crop), expected)


def test_pca_uses_physical_cloud_and_right_handed_axes():
    points = np.array([[x, y, z] for x in range(-3, 4) for y in (-1, 0, 1) for z in (-1, 0, 1)], float)
    points[:, 1] *= 5
    center, axes = _principal_axes(points)
    np.testing.assert_allclose(center, 0, atol=1e-12)
    assert abs(axes[0, 1]) == pytest.approx(1)
    np.testing.assert_allclose(axes @ axes.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(axes) == pytest.approx(1)


def test_translated_identical_molars_have_known_width(phantom):
    result = measure_widths(phantom[0])
    assert result.maxilla_width_mm == pytest.approx(40, abs=0.05)
    assert result.mandible_width_mm == pytest.approx(40, abs=0.05)
    assert all(t.rule_used != 'none' for t in result.per_tooth.values())


@pytest.mark.parametrize('transform', ['reflection', 'rotation', 'translation'])
def test_patient_space_width_invariance(phantom, tmp_path, transform):
    path, image = phantom
    reference = measure_widths(path)
    mapping = np.eye(4)
    if transform == 'reflection':
        mapping[0, 0] = -1
    elif transform == 'rotation':
        angle = np.deg2rad(23)
        mapping[:3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    else:
        mapping[:3, 3] = [120, -35, 8]
    changed = tmp_path / f'{transform}.nii.gz'
    nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), mapping @ image.affine), changed)
    measured = measure_widths(changed)
    for arch, width in reference.arch_widths_mm.items():
        assert measured.arch_widths_mm[arch] == pytest.approx(width, abs=0.25)


def test_missing_molar_is_reported_as_unmeasurable(phantom, tmp_path):
    _, image = phantom
    data = np.asanyarray(image.dataobj).copy()
    data[data == 14] = 0
    path = tmp_path / 'missing.nii.gz'
    nib.save(nib.Nifti1Image(data, image.affine), path)
    result = measure_widths(path)
    assert result.maxilla_width_mm is None
    assert result.per_tooth[14].confidence == 'low'
    assert result.per_tooth[14].rule_used == 'none'
