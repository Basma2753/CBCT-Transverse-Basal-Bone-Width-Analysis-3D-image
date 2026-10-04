import nibabel as nib
import pytest
from cbct_width.phantoms import make_phantom


@pytest.fixture
def phantom(tmp_path):
    path = tmp_path / 'synthetic_FULL_MASK.nii.gz'
    image = make_phantom()
    nib.save(image, path)
    return path, image
