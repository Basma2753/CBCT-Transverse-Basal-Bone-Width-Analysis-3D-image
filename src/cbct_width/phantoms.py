"""Deterministic toy tooth volumes for software tests and demonstrations.

These geometric shapes are not anatomically validated teeth or patient scans.
"""
import numpy as np
import nibabel as nib


def make_phantom():
    """Return four labelled toy molars on a 0.5 mm grid; both widths are 40 mm."""
    shape = (144, 104, 80)
    x, y, z = np.ogrid[:shape[0], :shape[1], :shape[2]]
    labels = np.zeros(shape, dtype=np.uint8)
    for label, cx, cy, roots in ((14, 28, 28, 3), (6, 108, 28, 3),
                                  (22, 28, 76, 2), (30, 108, 76, 2)):
        crown = ((x-cx)**2 + (y-cy)**2 <= 11**2) & (z >= 43) & (z <= 62)
        tooth = np.broadcast_to(crown, shape).copy()
        centers = ((-6, -4), (6, -4), (0, 7)) if roots == 3 else ((-6, 0), (6, 0))
        for dx, dy in centers:
            tooth |= ((x-cx-dx)**2 + (y-cy-dy)**2 <= 4**2) & (z >= 8) & (z < 44)
        labels[tooth] = label
    affine = np.diag([0.5, 0.5, 0.5, 1.0])
    return nib.Nifti1Image(labels, affine)
