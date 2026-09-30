"""
app.py  —  SELF-CONTAINED  (Colab-ready, single file)
=====================================================

Everything in one file: the tooth-segmentation orchestration, Dr. Nadeen's
furcation / centre-of-resistance width analysis, and the Streamlit UI.

There are NO local-module imports here (no `import pipeline`,
no `import molar_cr`) — Streamlit only needs this one script:

        streamlit run app.py

molar_cr is REVISION 2: the furcation scan waits for the tooth's full
anatomical root count (3 maxillary, 2 mandibular) instead of stopping
at the first 2 cross-sectional components, and the webbing region is
bounded by a true convex hull instead of a diamond-shaped closing.

Flow: upload one CBCT (.nii/.nii.gz) -> nnU-Net ToothFairy2 segmentation
-> molar_cr transverse-width analysis -> maxillary & mandibular widths (mm).
"""
from __future__ import annotations

# ==========================================================================
# SECTION 1 — molar_cr : furcation / centre-of-resistance analysis
# ==========================================================================
#  REVISION 2: exact root count (3 maxillary / 2 mandibular) with a
#             crown guard, + a true convex hull for the webbing
#             region. Ported from FINAL_DR_NADEEN_rev2.ipynb, with
#             the oblique-affine voxel-spacing fix kept.

"""
molar_cr.py
===========

Estimation of the Center of Resistance (CR) of first permanent molars from
3D CBCT segmentation masks, and computation of the transverse basal-bone
width as the Euclidean distance between bilateral CRs.

Scientific basis
----------------
The clinical convention followed here is the *Yonsei transverse analysis*
(Koo et al., Korean J Orthod 2017; 47:167-175), in which the geometric
centre of the root furcation of the first permanent molar is used as the
reference point for the tooth's centre of resistance. Zhang et al.
(AJODO 2023; 164:5-13) demonstrated that among the three main CBCT-based
transverse analyses (Yonsei, Penn, BU) the Yonsei method has the highest
inter-examiner reliability. The recent deep-learning study by Dai et al.
(BMC Oral Health 2024; 24:1091) also targets the furcation centre as the
CR proxy.

Biomechanical corroboration comes from Gandhi et al. (AJODO 2021;
160:442-450), who performed 3D finite-element analysis on 50 maxillary
first molars and localised the true biomechanical CR at 1.48 +/- 2.26 mm
distal and 0.19 +/- 1.75 mm apical to the trifurcation. The furcation
centre is therefore a well-justified low-variance surrogate. Viecilli et
al. (AJODO 2013; 143:163-172) and Dathe et al. (J Dent Biomech 2013;
4:1758736013499770) further show that a strict 3D CR "point" does not
exist for a geometrically asymmetric tooth; three non-intersecting axes
of resistance define a small CR volume. The furcation-centre estimate is
consistent with this reality and is what any subsequent biomechanical
model would be anchored to.

Algorithm outline
-----------------
For every labelled first molar:

    1. Extract the voxelwise binary tooth mask.
    2. Convert every foreground voxel to physical (mm) coordinates using
       the NIfTI affine (nibabel.affines.apply_affine). This correctly
       handles anisotropic voxel spacing, obliquity, and orientation.
    3. Principal Component Analysis (PCA) of the physical coordinates
       yields three orthonormal axes; the largest-eigenvalue axis v1 is
       the tooth's long (crown->root) direction. Doing PCA in physical
       space -- not voxel space -- is essential; otherwise anisotropic
       spacing biases the recovered direction.
    4. Sweep along v1. At each level t along v1, project the voxels near
       that plane onto the (v2, v3) coordinates, rasterise on a regular
       grid at the finest voxel spacing, and count connected components.
    5. The end of the tooth whose slices persistently contain a single
       component is the *crown*; the opposite end is the *root complex*.
    6. Moving from crown to root, the first level at which the number of
       components is >=2 for `persist` consecutive slices is the furcation
       plane.
    7. The furcation point is the geometric centre of the (v2, v3) region
       lying between (and touching) the roots at that plane, back-projected
       to patient coordinates.
    8. A 3D morphological skeletonisation cross-check reports the nearest
       skeleton branch point; large disagreement raises a low-confidence
       flag.

Bilateral distances are then Euclidean distances between the two furcation
points of the paired molars, in millimetres.

Author notes
------------
- No parameter in this module was tuned against the ground truth widths;
  all thresholds are derived from voxel spacing or set to values that are
  standard in the connected-component / persistence literature.
- The implementation is fully deterministic: eigenvector sign conventions
  are fixed, and any tie-breaks (e.g. equal-sized components) are broken
  by lexicographic ordering.
- Dependencies: numpy, scipy, scikit-image, nibabel. sklearn is not used.

Revision 2 -- arch-aware root count
-----------------------------------
Revision 1 declared the furcation at the first level, walking from the
crown, at which >= 2 cross-sectional components persisted. That rule is
correct for a two-rooted tooth and wrong for a three-rooted one, and it
has no guard against the crown.

  * The maxillary first permanent molar has THREE roots (mesiobuccal,
    distobuccal, palatal); the mandibular first permanent molar has TWO
    (mesial, distal). In a maxillary molar the furcation entrances open
    at different heights -- mesial most coronal, then buccal, then
    distal (Gher & Dunlap, J Periodontol 1985; 56:39-43). Walking
    apically, the horizontal cross-section therefore passes through a
    zone in which the MB root has separated but DB and P are still
    joined across the distal isthmus: exactly TWO components, neither of
    which is a single root. The webbing centroid at that level lies
    between the isolated buccal root and the mixed DB+P mass, i.e.
    displaced buccally from the true trifurcation centre. Both CRs move
    buccally, so the bilateral width is over-estimated -- consistent
    with the observed maxillary bias (+1.73 mm) being roughly double the
    mandibular one (+0.92 mm), the mandibular roots being arranged
    mesio-distally rather than bucco-palatally.

  * Nothing stopped the walk from declaring a "furcation" on the
    occlusal table, where the cusps rasterise as separate components
    over 2-3 mm -- longer than the `persist` window of 1.6 mm.

Revision 2 therefore:

  1. Takes the expected root count per arch (3 maxillary, 2 mandibular)
     and declares the furcation at the first level where EXACTLY that
     many components persist.
  2. Starts the crown->root walk at the crown's greatest cross-section
     (the height of contour). A furcation lies in the root complex,
     several mm apical to the CEJ, which is itself apical to the height
     of contour; so this restriction cannot exclude a true furcation,
     while it excludes every occlusal-cusp artefact.
  3. Derives crown-vs-root polarity from which end the area maximum is
     nearer to, rather than from the mean area of the outermost few
     slices (cusp tips and root apices are both small, so the old
     statistic could invert).
  4. Reports which rule fired for each tooth, so a relaxation is visible
     in the output rather than silent.

Neither 3 nor 2 is fitted to the ground-truth widths: they are the
anatomical root counts of the teeth being measured.
"""

import dataclasses as _dc
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import nibabel.processing as nibproc
from scipy import ndimage as ndi
from skimage.morphology import skeletonize as _skeletonize_nd  # 2D and 3D
from skimage.morphology import convex_hull_image as _convex_hull_image


# --------------------------------------------------------------------------- #
#  Data classes
# --------------------------------------------------------------------------- #

@_dc.dataclass
class ToothCR:
    """Per-tooth result.

    Attributes
    ----------
    label : int
        FDI-style label of the tooth in the segmentation mask.
    furcation_mm : np.ndarray, shape (3,)
        Yonsei-convention CR (furcation centre) in patient/world mm
        coordinates.
    long_axis_mm : np.ndarray, shape (3,)
        Unit vector of the tooth's long axis (crown->root sign), in
        patient mm coordinates.
    centroid_mm : np.ndarray, shape (3,)
        Volumetric centroid of the tooth in patient mm coordinates.
    skeleton_branch_mm : np.ndarray or None
        Coordinates of the coronal-most 3D skeleton branch point closest
        to the furcation estimate. Used as a cross-check.
    disagreement_mm : float
        Distance between the two estimates. Values >~2 mm mean the tooth
        should be visually reviewed.
    n_voxels : int
        Foreground voxel count of the tooth.
    confidence : str
        'high' if the two methods agree within 2 mm and a stable
        furcation transition was found; otherwise 'low'.
    n_roots_expected : int
        Anatomical root count declared for this tooth (3 for a maxillary
        first molar, 2 for a mandibular one).
    n_roots_found : int
        Number of cross-sectional components actually present at the
        chosen furcation plane. Equals ``n_roots_expected`` whenever the
        strict rule fired.
    rule_used : str
        Which rung of the root-count ladder produced the estimate:
        ``'exact:N'``   -- exactly N components persisted (the intended
                           case; the CR sits at the true N-way furcation);
        ``'at_least:N'``-- >= N persisted, so a spurious extra component
                           was tolerated;
        ``'relaxed:>=2'``-- the tooth never showed N separated roots
                           (fused/unresolved roots in the segmentation),
                           so revision-1 behaviour was used as a fallback.
                           These teeth carry the old buccal bias and
                           should be reviewed;
        ``'none'``      -- no furcation found.
    furcation_depth_mm : float
        Distance along the long axis from the crown's height of contour
        to the furcation plane. Anatomically this should be roughly
        6-11 mm for a first molar; a value under ~3 mm means the estimate
        is still sitting in the crown and is not trustworthy.
    """
    label: int
    furcation_mm: np.ndarray
    long_axis_mm: np.ndarray
    centroid_mm: np.ndarray
    skeleton_branch_mm: Optional[np.ndarray]
    disagreement_mm: float
    n_voxels: int
    confidence: str
    n_roots_expected: int = 0
    n_roots_found: int = 0
    rule_used: str = 'none'
    furcation_depth_mm: float = float('nan')


@_dc.dataclass
class CaseResult:
    """Per-CBCT result.

    ``arch_widths_mm`` is a dict mapping arch name (e.g. ``"maxilla"``,
    ``"mandible"``) to the transverse width in mm, or ``None`` if the
    arch's two teeth could not both be measured. The ``maxilla_width_mm``
    and ``mandible_width_mm`` attributes are convenience aliases into
    this dict.
    """
    case_id: str
    per_tooth: dict[int, ToothCR]
    arch_widths_mm: dict[str, Optional[float]]

    @property
    def maxilla_width_mm(self) -> Optional[float]:
        return self.arch_widths_mm.get("maxilla")

    @property
    def mandible_width_mm(self) -> Optional[float]:
        return self.arch_widths_mm.get("mandible")

    def as_row(self) -> dict:
        row = {"case_id": self.case_id}
        for arch, w in self.arch_widths_mm.items():
            row[f"{arch}_mm"] = w
        return row


# --------------------------------------------------------------------------- #
#  Low-level geometry
# --------------------------------------------------------------------------- #

def _apply_affine(coords_ijk: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Transform voxel-index coordinates to patient (mm) coordinates.

    Parameters
    ----------
    coords_ijk : (N, 3) array of voxel indices (i, j, k) as floats.
    affine : (4, 4) NIfTI affine matrix.

    Returns
    -------
    coords_mm : (N, 3) array of patient mm coordinates.

    Uses the standard nibabel convention `p = A @ [i, j, k, 1]^T`. This
    correctly propagates anisotropic voxel spacing, image obliquity, and
    the orientation encoded in the header.
    """
    return nib.affines.apply_affine(affine, coords_ijk.astype(np.float64))


def _principal_axes(points_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mean and the three principal-axis unit vectors of a
    point cloud, ordered from largest to smallest eigenvalue.

    Parameters
    ----------
    points_mm : (N, 3) array of physical coordinates in mm.

    Returns
    -------
    mean : (3,) centroid.
    axes : (3, 3) matrix whose ROWS are v1, v2, v3 (largest -> smallest
           variance). Sign convention: v1[k]>=0 if the axis-3 (usually
           superior-inferior) component is non-negative; this is arbitrary
           but deterministic, so re-runs give identical results.

    Notes
    -----
    Doing PCA in physical space rather than voxel-index space matters
    because anisotropic voxels would otherwise tilt v1 toward the
    finer-sampled axis. The covariance matrix is exact; no dimensionality
    reduction library is required, and no numerical randomness is
    introduced.
    """
    mean = points_mm.mean(axis=0)
    centred = points_mm - mean
    cov = np.cov(centred.T)                # (3, 3)
    eig_vals, eig_vecs = np.linalg.eigh(cov)   # ascending eigenvalues
    # eigh returns columns as eigenvectors; sort descending.
    order = np.argsort(eig_vals)[::-1]
    axes = eig_vecs[:, order].T            # rows = v1, v2, v3

    # Deterministic sign: for each eigenvector, force its largest-magnitude
    # component to be non-negative. Using a *fixed* index (e.g., v1[2] >= 0)
    # produces unstable signs when the tooth's long axis is nearly parallel
    # to a coordinate axis, because the other components are then noisy
    # near-zero values. Using the argmax-magnitude component eliminates that
    # instability. The crown-vs-root orientation of v1 is decided
    # downstream from the mask topology, not from this arbitrary sign.
    for r in range(3):
        k = int(np.argmax(np.abs(axes[r])))
        if axes[r, k] < 0:
            axes[r] = -axes[r]
    # Enforce right-handedness so downstream code has a consistent frame.
    if np.dot(np.cross(axes[0], axes[1]), axes[2]) < 0:
        axes[2] = -axes[2]
    return mean, axes


# --------------------------------------------------------------------------- #
#  Mask cleaning
# --------------------------------------------------------------------------- #

def _clean_tooth_mask(mask: np.ndarray, min_component_vox: int = 50) -> np.ndarray:
    """Fill interior holes and drop small speckles.

    Rationale: ToothSeg segmentations are reported as highly accurate but
    occasional interior pinholes (from partial volume near the pulp
    chamber) and isolated speckles from neighbouring teeth do occur.
    Filling holes is topologically safe; removing tiny components below
    ~50 voxels (~0.4 mm^3 at 0.2 mm voxels) prevents them from perturbing
    PCA or the connected-component count.
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return m
    m = ndi.binary_fill_holes(m)
    lbl, n = ndi.label(m, structure=np.ones((3, 3, 3), int))
    if n > 1:
        sizes = np.bincount(lbl.ravel())
        keep = np.zeros(n + 1, bool)
        keep[1:] = sizes[1:] >= min_component_vox
        # Keep only the largest connected component -- teeth are single
        # objects. Any secondary component of the same label is a
        # segmentation artefact.
        largest = 1 + int(np.argmax(sizes[1:]))
        keep[:] = False
        keep[largest] = True
        m = keep[lbl]
    return m


# --------------------------------------------------------------------------- #
#  Cross-section rasterisation and run counting
# --------------------------------------------------------------------------- #

def _cross_section_component_count(
    s2: np.ndarray,           # (M,) coordinate along v2 (mm)
    s3: np.ndarray,           # (M,) coordinate along v3 (mm)
    grid_step_mm: float,
    voxel_footprint_mm: float,
    min_component_area_mm2: float,
) -> tuple[int, np.ndarray, tuple[float, float]]:
    """Rasterise a point cloud onto a 2D grid and label components.

    Each voxel is stamped as a small disk of radius ~half the largest
    voxel dimension, so its raster footprint reflects the voxel's true
    physical extent. This is essential when the voxel spacing
    perpendicular to the tooth's long axis is coarser than the raster
    grid step: with single-pixel stamping, adjacent voxels within one
    root become disconnected in the raster and either fragment below the
    size threshold or require dilation large enough to bridge to the
    other root. Physically correct stamping makes the root
    cross-section a dense blob while preserving the multi-millimetre
    inter-root gap.

    Parameters
    ----------
    voxel_footprint_mm : diameter of the disk stamped for each voxel.
        Set to the max of the voxel spacings perpendicular to v1 (or the
        max of all spacings as a safe upper bound).
    min_component_area_mm2 : minimum physical area (mm^2) of a kept
        component. Converted internally to pixels via grid_step_mm.

    Returns
    -------
    n_components : int
    labels : (H, W) int array of the compact component labels.
    origin_mm : (v2_min, v3_min) so callers can back-project centroids.
    """
    if s2.size == 0:
        return 0, np.zeros((0, 0), int), (0.0, 0.0)

    v2_min, v2_max = s2.min(), s2.max()
    v3_min, v3_max = s3.min(), s3.max()

    # Padding: at least the stamping radius plus a couple of pixels.
    stamp_radius_pix = max(1, int(np.ceil(voxel_footprint_mm / 2.0
                                          / grid_step_mm)))
    pad = stamp_radius_pix + 2

    H = int(np.ceil((v3_max - v3_min) / grid_step_mm)) + 2 * pad + 1
    W = int(np.ceil((v2_max - v2_min) / grid_step_mm)) + 2 * pad + 1
    img = np.zeros((H, W), bool)
    jj = np.clip(((s2 - v2_min) / grid_step_mm).astype(int) + pad, 0, W - 1)
    ii = np.clip(((s3 - v3_min) / grid_step_mm).astype(int) + pad, 0, H - 1)
    img[ii, jj] = True

    # Stamp: dilate by a disk of radius = half the voxel footprint. This
    # is a physical extent, not a smoothing parameter. The stamping
    # radius is smaller than any realistic inter-root gap so it cannot
    # spuriously merge roots.
    if stamp_radius_pix >= 1:
        # A disk structure element.
        y, x = np.ogrid[-stamp_radius_pix:stamp_radius_pix + 1,
                        -stamp_radius_pix:stamp_radius_pix + 1]
        disk = (x * x + y * y) <= stamp_radius_pix * stamp_radius_pix
        img = ndi.binary_dilation(img, structure=disk)

    lbl, n = ndi.label(img, structure=np.ones((3, 3), int))
    if n == 0:
        return 0, lbl, (v2_min - pad * grid_step_mm,
                        v3_min - pad * grid_step_mm)

    sizes = np.bincount(lbl.ravel())
    min_component_pix = max(1, int(np.round(min_component_area_mm2
                                            / (grid_step_mm ** 2))))
    keep = sizes >= min_component_pix
    keep[0] = False                     # background never counts
    # Relabel to compact 1..K on kept components.
    remap = np.zeros_like(sizes)
    k = 0
    for old in range(1, len(sizes)):
        if keep[old]:
            k += 1
            remap[old] = k
    lbl = remap[lbl]
    return int(k), lbl, (v2_min - pad * grid_step_mm,
                         v3_min - pad * grid_step_mm)


def _convex_hull_2d(fg: np.ndarray) -> np.ndarray:
    """Exact convex hull of a 2D binary image.

    Rotation-independent and parameter-free, unlike a morphological
    closing with a fixed structuring element (see the comment at the call
    site). Falls back to the input if the hull is degenerate -- e.g. a
    cross-section whose pixels happen to be collinear -- which QHull
    cannot triangulate.
    """
    try:
        return _convex_hull_image(fg)
    except Exception:
        return fg


# --------------------------------------------------------------------------- #
#  Furcation detection (physical-space)
# --------------------------------------------------------------------------- #

# Root counts of the first permanent molars. These are anatomy, not tuning
# parameters: the maxillary first molar is tri-rooted (mesiobuccal,
# distobuccal, palatal); the mandibular first molar is bi-rooted (mesial,
# distal).
N_ROOTS_MAXILLARY_FIRST_MOLAR = 3
N_ROOTS_MANDIBULAR_FIRST_MOLAR = 2


def _find_furcation_3d(
    mask: np.ndarray,
    affine: np.ndarray,
    n_roots: int,
    step_mm: float = 0.4,
    persist: int = 4,
    min_component_area_mm2: float = 0.5,
    root_count_rule: str = 'exact',
    crown_polarity: str = 'area_peak',
    hull_method: str = 'convex',
    allow_relaxation: bool = True,
) -> dict:
    """3D furcation-centre estimation.

    Parameters
    ----------
    mask : boolean 3D array. Foreground = tooth.
    affine : (4, 4) NIfTI affine.
    n_roots : the tooth's anatomical root count -- the number of separated
        cross-sectional components that *defines* its furcation. 3 for a
        maxillary first molar, 2 for a mandibular one. Required: there is
        no sensible default, and silently assuming 2 is what biased
        revision 1's maxillary measurements buccally.
    root_count_rule : 'exact' (default) requires exactly ``n_roots``
        persistent components -- no fewer (which would catch a partial
        split where two roots are still joined) and no more (which would
        mean a spurious component). 'at_least' requires >= n_roots.
    crown_polarity : 'area_peak' (default) identifies the crown end as
        whichever end of the long axis the single largest cross-section
        lies nearer to. 'end_area' restores the revision-1 statistic
        (mean area of the outermost `persist` slices at each end), which
        is fragile because cusp tips and root apices are both small.
    hull_method : 'convex' (default) uses the exact convex hull of the
        furcation cross-section when locating the webbing centroid.
        'closing' restores revision 1's diamond-shaped morphological
        closing, which is orientation-dependent (see the comment at the
        call site). Retained for A/B comparison only.
    allow_relaxation : if True, a tooth whose roots never resolve into
        ``n_roots`` separate components (fused or under-segmented roots)
        falls back down the ladder rather than returning nothing. The
        rung used is always reported in the result.
    step_mm : slice thickness along the long axis, in mm. 0.4 mm is well
        below the diameter of a molar root (~1.5-3 mm at the neck) so
        the transition can be localised precisely, while being coarse
        enough to average out single-voxel noise.
    persist : require >=2 components for this many consecutive slices.
        Prevents a single noisy slice from being called a furcation.
        4 slices at 0.4 mm = 1.6 mm of continuous split, which is
        physiologically the minimum trunk-to-root separation.
    min_component_area_mm2 : minimum physical area of a kept 2D
        component. 0.5 mm^2 is well below any real molar root
        cross-section (typically 4-8 mm^2 at the neck) and above the
        area of a few isolated noise voxels. Specifying this in mm^2
        (not pixels) makes the algorithm invariant to raster resolution.

    Returns
    -------
    dict with keys
        'furcation_mm'   : (3,) coord in patient mm, or None if not found.
        'centroid_mm'    : (3,) tooth centroid in patient mm.
        'axes_mm'        : (3, 3) rows v1, v2, v3, unit vectors in patient mm.
        'long_axis_mm'   : (3,) unit vector, oriented crown->root.
        'profile'        : list of (t, n_components) along v1.
        'crown_at_low_t' : bool. True if crown is at low t along v1.
        'n_voxels'       : int
        'rule_used'      : str, which rung of the root-count ladder fired.
        'n_roots_found'  : int, components present at the chosen plane.
        'furcation_depth_mm' : float, distance from the height of contour
                           to the furcation plane along the long axis.
    """
    if int(n_roots) < 2:
        raise ValueError(f"n_roots must be >= 2; got {n_roots!r}.")
    if root_count_rule not in ('exact', 'at_least'):
        raise ValueError(
            f"root_count_rule must be 'exact' or 'at_least'; "
            f"got {root_count_rule!r}."
        )
    if crown_polarity not in ('area_peak', 'end_area'):
        raise ValueError(
            f"crown_polarity must be 'area_peak' or 'end_area'; "
            f"got {crown_polarity!r}."
        )
    if hull_method not in ('convex', 'closing'):
        raise ValueError(
            f"hull_method must be 'convex' or 'closing'; got {hull_method!r}."
        )
    m = _clean_tooth_mask(mask)
    n_vox = int(m.sum())
    if n_vox < 200:
        return {'furcation_mm': None, 'centroid_mm': None,
                'axes_mm': None, 'long_axis_mm': None,
                'profile': [], 'crown_at_low_t': True, 'n_voxels': n_vox,
                'rule_used': 'none', 'n_roots_found': 0,
                'furcation_depth_mm': float('nan')}

    ijk = np.column_stack(np.nonzero(m)).astype(np.float64)
    mm = _apply_affine(ijk, affine)
    mean, axes = _principal_axes(mm)
    v1, v2, v3 = axes

    # Project.
    centred = mm - mean
    t = centred @ v1
    s2 = centred @ v2
    s3 = centred @ v3

    # Slice along v1.
    # Adaptive step: the projection of the largest voxel dimension onto
    # v1 sets a lower bound. Using a step finer than that produces
    # aliasing (empty slices, apparent disconnections) in anisotropic
    # volumes -- the failure mode observed in early tests. We take the
    # projection of the voxel-size vector onto |v1| as the effective
    # sampling interval along v1 and require step_mm >= that.
    # Column norms, not abs(diag): a converted DICOM can carry an
    # oblique or axis-permuted affine, where the diagonal is not the
    # voxel size. For an axis-aligned affine the two agree.
    vox_sizes = np.linalg.norm(affine[:3, :3], axis=0)
    step_axis_mm = float(np.abs(v1) @ vox_sizes)   # projection onto v1
    step_mm_eff = max(step_mm, step_axis_mm * 1.25)
    t_min, t_max = float(t.min()), float(t.max())
    levels = np.arange(t_min, t_max + step_mm_eff, step_mm_eff)
    step_mm = step_mm_eff    # so all downstream code uses the effective step
    n_lev = len(levels)
    # For each voxel find its slice index.
    idx = np.clip(((t - t_min) / step_mm).astype(int), 0, n_lev - 1)

    # Cross-section rasterisation grid: subvoxel to preserve topology.
    grid_step = float(min(vox_sizes) * 0.75)  # subvoxel to avoid aliasing
    grid_step = max(grid_step, 0.05)          # sanity floor
    # Physical footprint stamped per voxel. Using max(vox_sizes) is a
    # safe upper bound valid for any orientation of v2, v3 and ensures
    # within-root densification is achieved without over-merging.
    voxel_footprint = float(max(vox_sizes))

    profile: list[tuple[float, int]] = []
    areas: list[float] = []            # foreground area (mm^2) per slice
    labels_at: dict[int, np.ndarray] = {}
    origins_at: dict[int, tuple[float, float]] = {}
    s2_at: dict[int, np.ndarray] = {}
    s3_at: dict[int, np.ndarray] = {}
    for i, tl in enumerate(levels):
        sel = (idx == i)
        n_pts_in_slab = int(sel.sum())
        s2_i = s2[sel]
        s3_i = s3[sel]
        n_c, lbl, origin = _cross_section_component_count(
            s2_i, s3_i, grid_step, voxel_footprint, min_component_area_mm2
        )
        profile.append((float(tl), int(n_c)))
        # Area proxy: number of foreground voxels in the slab, times the
        # in-slice mm^2 per voxel (approximated by voxel volume / slab
        # thickness). We only use this monotonically (crown = large area,
        # root = small area), so exact units are irrelevant.
        areas.append(float(n_pts_in_slab))
        labels_at[i] = lbl
        origins_at[i] = origin
        s2_at[i] = s2_i
        s3_at[i] = s3_i

    # ------------------------------------------------------------------ #
    #  Crown / root polarity
    # ------------------------------------------------------------------ #
    # Determine which end is the crown using CROSS-SECTIONAL AREA, not
    # component count. Component count alone is unreliable because (a)
    # very thin root tips can drop below the min-size threshold and
    # register as 0 or 1 components -- indistinguishable from the crown --
    # and (b) crown occlusal cusps artefactually split into multiple
    # components at the tip.
    #
    # 'area_peak' (default): the single largest cross-section of any molar
    # is the crown's height of contour, which sits in the coronal third of
    # the tooth. Whichever end of the long axis it is nearer to is
    # therefore the crown. This is a rank statistic over the whole
    # profile.
    #
    # 'end_area' (revision 1): compares the mean area of the outermost
    # `persist` slices at each end. Retained for A/B comparison only. It
    # is fragile because both ends are small there -- cusp tips at one
    # end, root apices at the other -- so the comparison can invert on a
    # tooth with short roots or worn cusps.
    non_empty = [i for i, (_, n_c) in enumerate(profile)
                 if n_c > 0 or areas[i] > 0]
    if len(non_empty) < 2 * persist:
        crown_at_low_t = True
    elif crown_polarity == 'area_peak':
        peak_pos = int(np.argmax([areas[i] for i in non_empty]))
        crown_at_low_t = peak_pos <= (len(non_empty) - 1 - peak_pos)
    else:
        head_area = float(np.mean([areas[i] for i in non_empty[:persist]]))
        tail_area = float(np.mean([areas[i] for i in non_empty[-persist:]]))
        crown_at_low_t = head_area >= tail_area

    # Order: crown -> root.
    order = non_empty if crown_at_low_t else non_empty[::-1]

    # ------------------------------------------------------------------ #
    #  Guard: the search may not begin in the crown
    # ------------------------------------------------------------------ #
    # The occlusal table of a molar rasterises as several disconnected
    # components (4 cusps maxillary, 5 mandibular) over the 2-3 mm from
    # the cusp tips down to where they coalesce. That is longer than the
    # `persist` window (1.6 mm), so ANY component-count rule applied over
    # the whole tooth can declare a "furcation" on the occlusal surface --
    # roughly 10 mm coronal to the real one. Revision 1 had no guard
    # against this.
    #
    # Starting the walk at the height of contour removes the failure mode
    # by construction: a furcation lies in the root complex, several mm
    # apical to the CEJ, which is itself apical to the height of contour.
    # No true furcation can be excluded; every cusp artefact is.
    ordered_areas = [areas[i] for i in order]
    hoc_rel = int(np.argmax(ordered_areas))       # height of contour
    search = order[hoc_rel:]

    # ------------------------------------------------------------------ #
    #  Root-count rule
    # ------------------------------------------------------------------ #
    def _first_persistent(predicate) -> Optional[int]:
        """First level in `search` whose next `persist` slices all satisfy
        `predicate` on the component count. Returns the slice index into
        `profile`, or None."""
        for k, i in enumerate(search):
            window = search[k:k + persist]
            if len(window) < persist:
                return None
            if all(predicate(profile[j][1]) for j in window):
                return i
        return None

    # Ladder, strictest first. Walking crown->root from the height of
    # contour, a clean tri-rooted molar's component count runs
    # 1,...,1,2,...,2,3,...,3 -- the 2-zone being the interval in which
    # the mesial and buccal furcation entrances have opened but the distal
    # isthmus still joins the DB and palatal roots. Stopping at the 2-zone
    # (revision 1) places the CR between an isolated buccal root and the
    # mixed DB+P mass, i.e. buccal of the true trifurcation centre.
    # Requiring exactly n_roots stops at the first level where all three
    # roots are genuinely separated -- the trifurcation roof.
    ladder: list[tuple[str, object]] = []
    if root_count_rule == 'exact':
        ladder.append((f'exact:{n_roots}', lambda c: c == n_roots))
    ladder.append((f'at_least:{n_roots}', lambda c: c >= n_roots))
    if n_roots > 2:
        # Last resort: roots that the segmentation never resolves into
        # n_roots separate components (fused roots, or partial volume
        # across the narrow MB-DB gap). Revision-1 behaviour, kept so the
        # tooth still yields a number -- but reported, because it carries
        # the revision-1 buccal bias.
        ladder.append(('relaxed:>=2', lambda c: c >= 2))
    if not allow_relaxation:
        ladder = ladder[:1]          # strictest rung only; fail otherwise

    furc_slice_idx = None
    rule_used = 'none'
    for name, predicate in ladder:
        furc_slice_idx = _first_persistent(predicate)
        if furc_slice_idx is not None:
            rule_used = name
            break

    long_axis_signed = v1 if crown_at_low_t else -v1

    if furc_slice_idx is None:
        return {'furcation_mm': None, 'centroid_mm': mean,
                'axes_mm': axes, 'long_axis_mm': long_axis_signed,
                'profile': profile, 'crown_at_low_t': crown_at_low_t,
                'n_voxels': n_vox, 'rule_used': 'none', 'n_roots_found': 0,
                'furcation_depth_mm': float('nan')}

    n_roots_found = int(profile[furc_slice_idx][1])
    furcation_depth_mm = float(abs(profile[furc_slice_idx][0]
                                   - profile[order[hoc_rel]][0]))

    # Localise the furcation *point*. At the furcation slice, the mask
    # has >=2 components. The Yonsei "centre of furcation" is the point
    # in the roof of the split region -- i.e., the centroid of the
    # smallest region that spans between the components.
    #
    # Approach: take the convex hull of all foreground pixels at the
    # furcation slice, subtract the labelled components, and take the
    # centroid of what remains (the "webbing" between the roots). If the
    # remaining region is empty (roots already fully separated), use the
    # centroid of the pairwise-nearest boundary segments.
    tl = profile[furc_slice_idx][0]
    lbl = labels_at[furc_slice_idx]
    origin = origins_at[furc_slice_idx]

    # Fill the convex hull of the union of components on this slice.
    fg = lbl > 0
    if fg.sum() < 4:
        # Degenerate; use plain foreground centroid.
        ii, jj = np.nonzero(fg)
        if ii.size == 0:
            centre_yx = (0.0, 0.0)
        else:
            centre_yx = (ii.mean(), jj.mean())
    else:
        # True convex hull of the cross-section.
        #
        # Revision 1 approximated the hull with `binary_closing(fg,
        # iterations=round(2.0/grid_step))`. scipy's default structuring
        # element is the 4-connected cross, so N iterations dilate by an
        # L1 ball of radius N -- a DIAMOND, whose reach along the raster
        # diagonals is 1/sqrt(2) of its reach along the axes. Its ability
        # to bridge an inter-root channel therefore varied by ~40 % with
        # the channel's orientation in the (v2, v3) frame, which PCA
        # fixes arbitrarily. With two components and one wide gap that is
        # harmless. With three roots there are three channels at three
        # different orientations, each near the diamond's bridging limit,
        # so the webbing region -- and its centroid -- became a function
        # of the tooth's incidental orientation. Measured on a mirrored
        # phantom pair, which must by symmetry give identical answers, the
        # closing produced a 1.1 mm discrepancy between the two sides.
        #
        # The convex hull is exact, deterministic, rotation-independent,
        # and removes the 2 mm radius (the module's only remaining
        # hand-set length scale in this step).
        if hull_method == 'convex':
            hull = _convex_hull_2d(fg)
        else:
            rr = int(round(2.0 / grid_step))    # revision-1 behaviour
            hull = ndi.binary_closing(fg, iterations=rr)
        webbing = hull & ~fg
        if webbing.sum() >= 4:
            ii, jj = np.nonzero(webbing)
            centre_yx = (ii.mean(), jj.mean())
        else:
            # Roots already fused visually at this slice: fall back to
            # centroid of the whole cross-section, which for a symmetric
            # multi-root tooth also approximates the trunk axis.
            ii, jj = np.nonzero(fg)
            centre_yx = (ii.mean(), jj.mean())

    s3_pt = origin[1] + centre_yx[0] * grid_step
    s2_pt = origin[0] + centre_yx[1] * grid_step

    furcation_mm = mean + tl * v1 + s2_pt * v2 + s3_pt * v3

    return {'furcation_mm': furcation_mm, 'centroid_mm': mean,
            'axes_mm': axes, 'long_axis_mm': long_axis_signed,
            'profile': profile, 'crown_at_low_t': crown_at_low_t,
            'n_voxels': n_vox, 'rule_used': rule_used,
            'n_roots_found': n_roots_found,
            'furcation_depth_mm': furcation_depth_mm}


# --------------------------------------------------------------------------- #
#  Skeleton cross-check
# --------------------------------------------------------------------------- #

def _skeleton_branch_points_mm(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """3D morphological skeleton branch points in patient mm.

    A skeleton voxel is a branch (junction) if it has 3 or more skeleton
    neighbours in a 3x3x3 window. For a molar these correspond to the
    canal-tree bifurcation and any cusp junctions on the crown. Used
    only as an independent check; not as primary estimator, because
    3D skeletonisation is noise-sensitive near thin structures.
    """
    m = _clean_tooth_mask(mask)
    sk = _skeletonize_nd(m.astype(np.uint8)).astype(bool)
    if not sk.any():
        return np.zeros((0, 3))
    neigh = ndi.convolve(sk.astype(np.int8), np.ones((3, 3, 3), np.int8),
                         mode='constant', cval=0) - 1
    branches = np.argwhere(sk & (neigh >= 3))
    if branches.size == 0:
        return np.zeros((0, 3))
    return _apply_affine(branches.astype(np.float64), affine)


# --------------------------------------------------------------------------- #
#  Per-tooth entry point
# --------------------------------------------------------------------------- #

def estimate_tooth_cr(
    tooth_mask: np.ndarray,
    affine: np.ndarray,
    label: int,
    n_roots: int,
    disagreement_threshold_mm: float = 2.0,
    **furcation_kwargs,
) -> ToothCR:
    """Estimate the CR (furcation-centre convention) of a single tooth.

    See module docstring for the algorithm. This function ties together
    the primary detection and the skeleton cross-check.

    Parameters
    ----------
    tooth_mask : boolean 3D array. True where this tooth is present.
    affine : (4, 4) NIfTI affine of the mask.
    label : the tooth's FDI label, propagated to the output.
    n_roots : the tooth's anatomical root count (3 maxillary first molar,
        2 mandibular). Required -- see ``_find_furcation_3d``.
    disagreement_threshold_mm : primary and skeleton estimates further
        apart than this trigger `confidence='low'` in the result.
    **furcation_kwargs : forwarded to ``_find_furcation_3d`` (e.g.
        ``root_count_rule``, ``crown_polarity``, ``allow_relaxation``).

    Returns
    -------
    ToothCR
    """
    result = _find_furcation_3d(tooth_mask, affine, n_roots=n_roots,
                                **furcation_kwargs)
    fp = result['furcation_mm']
    if fp is None:
        return ToothCR(
            label=label, furcation_mm=np.array([np.nan] * 3),
            long_axis_mm=(result['long_axis_mm']
                          if result['long_axis_mm'] is not None
                          else np.array([np.nan] * 3)),
            centroid_mm=(result['centroid_mm']
                         if result['centroid_mm'] is not None
                         else np.array([np.nan] * 3)),
            skeleton_branch_mm=None,
            disagreement_mm=float('nan'),
            n_voxels=result['n_voxels'],
            confidence='low',
            n_roots_expected=int(n_roots),
            n_roots_found=int(result.get('n_roots_found', 0)),
            rule_used=str(result.get('rule_used', 'none')),
            furcation_depth_mm=float(result.get('furcation_depth_mm',
                                                float('nan'))),
        )

    # Skeleton cross-check: closest branch point *on the root side* of
    # the tooth. We reject branch points on the crown side because
    # cusps of enamel produce spurious 3-way junctions in the 3D
    # skeleton, which are not comparable to the root furcation. The
    # root side is defined by the crown->root sign of `long_axis_mm`.
    branches_mm = _skeleton_branch_points_mm(tooth_mask, affine)
    disagreement = float('inf')
    nearest_branch = None
    if branches_mm.size:
        # Project branches onto the crown->root axis relative to the
        # furcation estimate. Positive = apical (root) side, negative =
        # coronal side. Keep only branches within a small coronal margin
        # (so a branch just at the furcation counts) and on the root side.
        rel = (branches_mm - fp[None, :]) @ result['long_axis_mm']
        mask_root_side = rel >= -1.0   # allow 1 mm coronal slack
        if mask_root_side.any():
            candidates = branches_mm[mask_root_side]
            d = np.linalg.norm(candidates - fp[None, :], axis=1)
            j = int(np.argmin(d))
            disagreement = float(d[j])
            nearest_branch = candidates[j]

    # Three-level confidence:
    #   'high'   -> skeleton branch found and agrees within threshold
    #   'medium' -> primary furcation found, no root-side skeleton branch
    #               available for cross-check (does not indicate a
    #               problem, only that the independent check is silent)
    #   'low'    -> skeleton branch found but disagrees by > threshold,
    #               OR the strict root-count rule did not fire (the tooth's
    #               roots never resolved into n_roots components, so the
    #               estimate carries the revision-1 bias)
    rule_used = str(result.get('rule_used', 'none'))
    strict_rule_fired = rule_used.startswith('exact:')
    if not strict_rule_fired:
        confidence = 'low'
    elif nearest_branch is None:
        confidence = 'medium'
    elif disagreement <= disagreement_threshold_mm:
        confidence = 'high'
    else:
        confidence = 'low'

    return ToothCR(
        label=label,
        furcation_mm=fp,
        long_axis_mm=result['long_axis_mm'],
        centroid_mm=result['centroid_mm'],
        skeleton_branch_mm=nearest_branch,
        disagreement_mm=disagreement,
        n_voxels=result['n_voxels'],
        confidence=confidence,
        n_roots_expected=int(n_roots),
        n_roots_found=int(result.get('n_roots_found', 0)),
        rule_used=rule_used,
        furcation_depth_mm=float(result.get('furcation_depth_mm',
                                            float('nan'))),
    )


# --------------------------------------------------------------------------- #
#  Case entry point
# --------------------------------------------------------------------------- #

# FDI labels for the four first permanent molars.
# 16 = maxillary right; 26 = maxillary left;
# 36 = mandibular left; 46 = mandibular right.
FDI_MOLARS = {
    'maxillary_right': 16,
    'maxillary_left':  26,
    'mandibular_left': 36,
    'mandibular_right': 46,
}

# Default arch label map: two labels per arch, plus the arch's root count.
# For every arch listed here, ``process_case`` measures the transverse
# width as the Euclidean distance between the CRs of that arch's two
# teeth. Labels are integer values in the segmentation volume; names are
# human-readable identifiers used for logging only. The third element is
# the anatomical root count of the arch's first molars -- 3 in the
# maxilla, 2 in the mandible. Left/right names carry the same handedness
# correction as ``DEFAULT_ARCH_MAPS`` (see its comment): in this
# pipeline's arrays the model's label 26 sits on the physical maxillary
# RIGHT first molar, 16 on the LEFT, 36 on the mandibular RIGHT and 46
# on the LEFT.
DEFAULT_ARCH_LABEL_MAPS: list[tuple[str, dict[int, str], int]] = [
    ("maxilla",  {26: "UR6", 16: "UL6"}, N_ROOTS_MAXILLARY_FIRST_MOLAR),
    ("mandible", {36: "LR6", 46: "LL6"}, N_ROOTS_MANDIBULAR_FIRST_MOLAR),
]


def _coerce_arch(spec) -> tuple[str, dict[int, str], int]:
    """Normalise an arch specification to ``(name, labels, n_roots)``.

    Revision-1 two-element specs are rejected rather than defaulted:
    silently assuming two roots is exactly the bug this revision fixes,
    and a silent wrong answer is worse than a loud error.
    """
    spec = tuple(spec)
    if len(spec) == 3:
        name, labels, n_roots = spec
        return str(name), dict(labels), int(n_roots)
    if len(spec) == 2:
        name = spec[0]
        raise ValueError(
            f"Arch {name!r} does not declare a root count. molar_cr "
            f"revision 2 requires (arch_name, label_map, n_roots), e.g.\n"
            f'    ("maxilla",  {{...}}, 3)   # first molars: MB, DB, palatal\n'
            f'    ("mandible", {{...}}, 2)   # first molars: mesial, distal'
        )
    raise ValueError(
        f"Arch spec must be (name, labels, n_roots); got {spec!r}."
    )


def _tooth_bounding_boxes(data, labels, margin_vox: int = 2) -> dict:
    """Find all requested tooth extents, retaining surrounding background.

    Integer segmentations use one C-level pass over the scan. Floating-point
    label files retain their previous exact-equality semantics via a fallback.
    Bounds include every component (even speckles): cleanup still makes the
    original largest-component decision. Margins are clipped at true scan
    edges, so cropping does not invent context outside the acquired volume.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected a 3-D segmentation; got shape {data.shape}.")
    labels = list(dict.fromkeys(labels))
    if not labels:
        return {}
    margin = max(1, int(margin_vox))
    standard_labels = (
        data.dtype.kind in "biu"
        and all(isinstance(lbl, (int, np.integer)) and 0 < lbl <= 4096
                for lbl in labels)
    )
    raw_boxes = {}
    if standard_labels:
        # max_label bounds the lookup table; unrelated labels are ignored.
        found = ndi.find_objects(data, max_label=int(max(labels)))
        raw_boxes = {lbl: found[int(lbl) - 1] for lbl in labels}
    else:
        for lbl in labels:
            mask = data == lbl
            extents = []
            for axis in range(3):
                other_axes = tuple(i for i in range(3) if i != axis)
                occupied = np.flatnonzero(np.any(mask, axis=other_axes))
                if occupied.size == 0:
                    extents = []
                    break
                extents.append(slice(int(occupied[0]), int(occupied[-1]) + 1))
            raw_boxes[lbl] = tuple(extents) if extents else None

    return {
        lbl: (
            tuple(slice(max(0, bound.start - margin),
                        min(data.shape[axis], bound.stop + margin))
                  for axis, bound in enumerate(box))
            if box is not None else None
        )
        for lbl, box in raw_boxes.items()
    }


def _affine_for_crop(affine: np.ndarray, bounds: tuple) -> np.ndarray:
    """Keep every cropped voxel at its original patient/world-mm location."""
    cropped = np.array(affine, dtype=np.float64, copy=True)
    origin_ijk = np.array([bound.start for bound in bounds], dtype=np.float64)
    cropped[:3, 3] = nib.affines.apply_affine(affine, origin_ijk)
    return cropped


def process_case(
    seg_path: str | Path,
    case_id: str | None = None,
    arch_label_maps: list[tuple[str, dict[int, str], int]] | None = None,
    **furcation_kwargs,
) -> CaseResult:
    """Full pipeline for one CBCT segmentation volume.

    Parameters
    ----------
    seg_path : path to a NIfTI segmentation volume where each tooth has
        a distinct integer label. The CBCT intensity image is not
        required for CR estimation; the geometry lives entirely in the
        mask + affine.
    case_id : identifier propagated to the result.
    arch_label_maps : list of ``(arch_name, {int_label: tooth_name},
        n_roots)`` tuples. Each arch's transverse width is the distance
        between the CRs of its two teeth; ``n_roots`` is the anatomical
        root count of those teeth (3 maxillary first molar, 2 mandibular)
        and defines what counts as that tooth's furcation. Defaults to
        ``DEFAULT_ARCH_LABEL_MAPS`` (FDI 16/26 for maxilla with 3 roots,
        46/36 for mandible with 2).
    **furcation_kwargs : forwarded to ``estimate_tooth_cr`` (e.g.
        ``root_count_rule='at_least'``, ``crown_polarity='end_area'``)
        for A/B comparison against revision 1.

    Returns
    -------
    CaseResult with per-tooth CRs and per-arch bilateral widths (mm).
    """
    arches = [_coerce_arch(a) for a in (
        arch_label_maps if arch_label_maps is not None
        else DEFAULT_ARCH_LABEL_MAPS
    )]
    seg = nib.load(str(seg_path))
    data = np.asanyarray(seg.dataobj)   # avoid float cast for large vols
    affine = seg.affine
    tooth_boxes = _tooth_bounding_boxes(
        data, [lbl for _arch, label_map, _roots in arches for lbl in label_map]
    )
    if case_id is None:
        case_id = Path(seg_path).stem

    per_tooth: dict[int, ToothCR] = {}
    arch_widths: dict[str, Optional[float]] = {}

    for arch_name, label_map, n_roots in arches:
        if len(label_map) != 2:
            raise ValueError(
                f"Arch {arch_name!r} must have exactly 2 labels for a "
                f"transverse width; got {len(label_map)}."
            )
        # Estimate CR for each tooth in the arch.
        for lbl, _tooth_name in label_map.items():
            if lbl in per_tooth:
                continue                # label already processed
            bounds = tooth_boxes[lbl]
            if bounds is None:
                per_tooth[lbl] = ToothCR(
                    label=lbl,
                    furcation_mm=np.array([np.nan] * 3),
                    long_axis_mm=np.array([np.nan] * 3),
                    centroid_mm=np.array([np.nan] * 3),
                    skeleton_branch_mm=None,
                    disagreement_mm=float('nan'),
                    n_voxels=0,
                    confidence='low',
                    n_roots_expected=n_roots,
                    rule_used='none',
                )
                continue
            # All full-resolution tooth voxels are retained. Only distant
            # background is excluded from repeated 3-D morphology.
            mask = data[bounds] == lbl
            crop_affine = _affine_for_crop(affine, bounds)
            per_tooth[lbl] = estimate_tooth_cr(mask, crop_affine, lbl,
                                               n_roots=n_roots,
                                               **furcation_kwargs)

        # Compute the arch width from its two teeth.
        lbls = list(label_map.keys())
        a = per_tooth[lbls[0]]
        b = per_tooth[lbls[1]]
        if np.any(np.isnan(a.furcation_mm)) or np.any(np.isnan(b.furcation_mm)):
            arch_widths[arch_name] = None
        else:
            arch_widths[arch_name] = float(
                np.linalg.norm(a.furcation_mm - b.furcation_mm)
            )

    return CaseResult(
        case_id=case_id,
        per_tooth=per_tooth,
        arch_widths_mm=arch_widths,
    )


# ==========================================================================
# SECTION 2 — pipeline : single-scan segmentation + measurement
# ==========================================================================
"""
pipeline.py
===========

Single-scan orchestration that chains the two stages of the project:

    raw CBCT (.nii / .nii.gz)
        │
        ▼   Stage 1  — nnU-Net v2 ToothFairy2 tooth segmentation
    per-tooth label mask (labels 1..32)
        │
        ▼   Stage 2  — molar_cr furcation / centre-of-resistance analysis
    transverse basal-bone widths (maxilla, mandible) in millimetres

This module contains no ground-truth handling. It is the inference-only
path used by the Streamlit interface (``app.py``): upload one scan, get the
predicted widths out.

The segmentation stage is a thin, single-case wrapper around the exact
nnU-Net invocation used in the original ``segmentation.ipynb`` batch
notebook. The measurement stage calls Dr. Nadeen's ``molar_cr`` module
unmodified.
"""


import os
import glob
import gzip
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import nibabel as nib



# --------------------------------------------------------------------------- #
#  Configuration
# --------------------------------------------------------------------------- #

# Public release of the ToothFairy2 winning tooth-segmentation model.
ZENODO_URL = "https://zenodo.org/records/14893540/files/ToothSeg.zip"
DATASET_NAME = "Dataset121_ToothFairy2_Teeth"
DATASET_ID = "121"

# Label -> tooth mapping for the FIRST PERMANENT MOLARS in this model's
# output, AFTER the left/right handedness correction.
#
# The ToothFairy2 model emits dense sequential labels 1..32 (8 teeth per
# quadrant, central incisor -> third molar), so the 6th tooth of each
# quadrant is the first molar -- but WHICH side the model calls "right"
# depends on the array handedness it was trained on. The DICOM -> NIfTI
# conversion (SimpleITK, LPS -> RAS) negates the patient's left-right and
# front-back axes in the affine, which flips the array handedness relative
# to the model's training convention: on the physical anatomy, label 6
# lands on the maxillary LEFT first molar, label 14 on the RIGHT, and
# likewise 22/30 in the mandible. Verified on the validation cohort by
# mapping ground-truth landmarks onto the segmentation with zero fitted
# parameters: every GT point lands inside the corrected label's tooth,
# and none lands inside the direct one.
#
# The corrected mapping is therefore:
#     14 = maxillary  right  first molar (UR6)
#     6  = maxillary  left   first molar (UL6)
#     30 = mandibular left   first molar (LL6)
#     22 = mandibular right  first molar (LR6)
# This is the mapping Dr. Nadeen validated against the clinical widths.
# The correction only renames left/right: the bilateral widths are
# unchanged (a distance does not care which side is which), and the GT
# landing check re-verifies the assignment on every validated case -- a
# scan converted by a different chain can arrive with the opposite
# handedness, in which case the app detects it and switches (see the
# ground-truth validation section).
# The third element is the arch's anatomical root count, required by
# molar_cr revision 2: a maxillary first molar has 3 roots (MB, DB,
# palatal), a mandibular one has 2 (mesial, distal). Neither number is
# fitted to anything; they are anatomy.
DEFAULT_ARCH_MAPS: list[tuple[str, dict[int, str], int]] = [
    ("maxilla",  {14: "UR6", 6: "UL6"}, N_ROOTS_MAXILLARY_FIRST_MOLAR),
    ("mandible", {22: "LR6", 30: "LL6"}, N_ROOTS_MANDIBULAR_FIRST_MOLAR),
]

# --------------------------------------------------------------------------- #
#  Diagnostic cut-offs
# --------------------------------------------------------------------------- #
#
#  (A) PRIMARY CLASSIFICATION — Yonsei Transverse Index (the DIFFERENCE).
#
#      index = maxillary transverse width - mandibular transverse width
#
#      Normal occlusion averages -0.39 +/- 1.87 mm, so the two cut-offs are
#      (mean - 1 SD) = -2.26 mm and (mean + 1 SD) = +1.48 mm:
#
#          index  <  -2.26 mm        -> skeletal crossbite
#          -2.26 <= index <= +1.48   -> normal transverse skeletal relationship
#          index  >  +1.48 mm        -> skeletal transverse excess pattern
#
YONSEI_CUTOFF_MM = -2.26          # lower cut-off (crossbite)
YONSEI_EXCESS_CUTOFF_MM = 1.48    # upper cut-off (transverse excess)
YONSEI_NORMAL_MEAN_MM = -0.39
YONSEI_NORMAL_SD_MM = 1.87

#  (B) SUB-CLASSIFICATION — absolute per-arch width.
#
#      The index above says whether the two arches MATCH; it cannot say WHICH
#      arch is at fault. A narrow maxilla over a narrow mandible gives a
#      "normal" index. These absolute norms sub-classify each arch on its own:
#
#          maxilla :  < 45.64 deficiency | 45.64-51.08 normal | > 51.08 excess
#          mandible:  < 46.30 deficiency | 46.30-51.20 normal | > 51.20 excess
#
ARCH_WIDTH_NORMS_MM: dict[str, tuple[float, float]] = {
    "maxilla":  (45.64, 51.08),
    "mandible": (46.30, 51.20),
}

# Adjectival form used in the printed sub-class label.
_ARCH_ADJECTIVE = {"maxilla": "Maxillary", "mandible": "Mandibular"}


def classify_transverse(maxilla_mm: float, mandible_mm: float,
                        crossbite_cutoff_mm: float = YONSEI_CUTOFF_MM,
                        excess_cutoff_mm: float = YONSEI_EXCESS_CUTOFF_MM,
                        cutoff_mm: Optional[float] = None) -> dict:
    """Compute the Yonsei Transverse Index and classify it three ways.

    Parameters
    ----------
    maxilla_mm          : maxillary transverse width (Mx-Mx), mm.
    mandible_mm         : mandibular transverse width (Md-Md), mm.
    crossbite_cutoff_mm : index BELOW this is crossbite. Default -2.26 mm.
    excess_cutoff_mm    : index ABOVE this is transverse excess. Default
                          +1.48 mm.
    cutoff_mm           : deprecated alias for `crossbite_cutoff_mm`, kept so
                          older calls keep working.

    Returns
    -------
    dict with:
        index_mm     : maxilla_mm - mandible_mm
        category     : 'crossbite' | 'normal' | 'excess'
        label        : human-readable classification
        is_crossbite : True if category == 'crossbite'  (legacy key)
        is_excess    : True if category == 'excess'
        cutoff_mm, crossbite_cutoff_mm, excess_cutoff_mm : thresholds used
    """
    if cutoff_mm is not None:              # backwards-compatible alias
        crossbite_cutoff_mm = cutoff_mm

    index = float(maxilla_mm) - float(mandible_mm)
    # Decide on the value rounded to the reported precision (0.01 mm) so the
    # category always matches the 2-decimal number shown to the user, and
    # never flips due to binary floating-point artefacts at an exact cut-off
    # (e.g. 27.74 - 30.0 not being exactly -2.26).
    idx = round(index, 2)

    if idx < crossbite_cutoff_mm:
        category = "crossbite"
        label = "Skeletal crossbite (maxillary transverse deficiency)"
    elif idx > excess_cutoff_mm:
        category = "excess"
        label = "Skeletal transverse excess pattern"
    else:
        category = "normal"
        label = "Normal transverse skeletal relationship"

    return {
        "index_mm": index,
        "category": category,
        "label": label,
        "is_crossbite": category == "crossbite",
        "is_excess": category == "excess",
        "cutoff_mm": crossbite_cutoff_mm,              # legacy key
        "crossbite_cutoff_mm": crossbite_cutoff_mm,
        "excess_cutoff_mm": excess_cutoff_mm,
    }


def classify_arch_width(width_mm: Optional[float], arch: str,
                        norms: Optional[dict] = None) -> dict:
    """Sub-classify ONE arch on its absolute transverse width.

    Parameters
    ----------
    width_mm : that arch's transverse width in mm, or None if unmeasured.
    arch     : 'maxilla' or 'mandible'.
    norms    : optional {arch: (lower_mm, upper_mm)} override. Defaults to
               ARCH_WIDTH_NORMS_MM.

    Returns
    -------
    dict with:
        arch, width_mm
        category : 'deficiency' | 'normal' | 'excess' | 'unknown'
        label    : e.g. 'Maxillary deficiency'
        lower_mm, upper_mm : the normal-range bounds used
    """
    key = str(arch).strip().lower()
    table = ARCH_WIDTH_NORMS_MM if norms is None else norms

    if width_mm is None or key not in table:
        return {"arch": key, "width_mm": None, "category": "unknown",
                "label": "-", "lower_mm": None, "upper_mm": None}

    lower, upper = table[key]
    w = round(float(width_mm), 2)          # same rounding rule as above
    if w < lower:
        category, word = "deficiency", "deficiency"
    elif w > upper:
        category, word = "excess", "excess"
    else:
        category, word = "normal", "normal width"

    adj = _ARCH_ADJECTIVE.get(key, key.capitalize())
    return {
        "arch": key,
        "width_mm": float(width_mm),
        "category": category,
        "label": f"{adj} {word}",
        "lower_mm": lower,
        "upper_mm": upper,
    }


def _noop(*_a, **_k):
    pass


# --------------------------------------------------------------------------- #
#  Model set-up / discovery
# --------------------------------------------------------------------------- #

def get_results_dir() -> str:
    """Directory nnU-Net looks in for trained models.

    Honours the ``nnUNet_results`` environment variable if the caller has
    already set it (the Colab launcher does); otherwise falls back to a
    sensible default under ``/content``.
    """
    return os.environ.get("nnUNet_results", "/content/nnUNet_results")


def model_is_ready(results_dir: Optional[str] = None) -> bool:
    """True if the ToothFairy2 dataset folder is present and populated."""
    results_dir = results_dir or get_results_dir()
    d = os.path.join(results_dir, DATASET_NAME)
    return os.path.isdir(d) and len(os.listdir(d)) > 0


def detect_model_config(results_dir: Optional[str] = None) -> tuple[str, str, str]:
    """Return (trainer, plans, config) parsed from the model subfolder name.

    nnU-Net stores a trained model under a folder named
    ``<trainer>__<plans>__<config>``. We read whatever the release shipped
    rather than hard-coding it, exactly as the original batch notebook did.
    """
    results_dir = results_dir or get_results_dir()
    dataset_dir = os.path.join(results_dir, DATASET_NAME)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Model folder not found: {dataset_dir}. Run setup_model() first."
        )
    subdirs = [
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and "__" in d
    ]
    if not subdirs:
        raise RuntimeError(
            f"No '<trainer>__<plans>__<config>' subfolder inside {dataset_dir}."
        )
    trainer, plans, config = subdirs[0].split("__")
    return trainer, plans, config


def setup_model(
    results_dir: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> tuple[str, str, str]:
    """Ensure the ToothFairy2 model is downloaded and correctly placed.

    Idempotent: if the model is already present this just detects and
    returns its (trainer, plans, config). Otherwise it downloads the
    ~GB release from Zenodo, unzips it, and moves the dataset folder into
    ``results_dir``.

    Returns
    -------
    (trainer, plans, config)
    """
    results_dir = results_dir or get_results_dir()
    os.makedirs(results_dir, exist_ok=True)
    os.environ["nnUNet_results"] = results_dir

    dataset_dir = os.path.join(results_dir, DATASET_NAME)

    if model_is_ready(results_dir):
        log("Segmentation model already installed.")
    else:
        log("Segmentation model not found — downloading from Zenodo "
            "(this is a one-time ~GB download)...")
        cwd = os.getcwd()
        try:
            os.chdir(results_dir)
            # -nc: skip if already downloaded; keeps re-runs cheap.
            os.system(f'wget -nc -q --show-progress "{ZENODO_URL}"')
            log("Unzipping model archive...")
            os.system("unzip -q -o ToothSeg.zip")
            matches = glob.glob(
                os.path.join(results_dir, "**", DATASET_NAME), recursive=True
            )
            if not matches:
                raise RuntimeError(
                    f"'{DATASET_NAME}' not found after extraction. "
                    f"Contents: {os.listdir(results_dir)}"
                )
            found = matches[0]
            if os.path.abspath(found) != os.path.abspath(dataset_dir):
                shutil.move(found, dataset_dir)
            log("Model installed successfully.")
        finally:
            os.chdir(cwd)

    trainer, plans, config = detect_model_config(results_dir)
    log(f"Model config -> trainer={trainer}  plans={plans}  config={config}")
    return trainer, plans, config


# --------------------------------------------------------------------------- #
#  Input staging
# --------------------------------------------------------------------------- #

def _stage_input(input_path: str | Path, input_dir: str | Path,
                 case_id: str) -> str:
    """Copy/convert the uploaded scan into the nnU-Net input folder.

    nnU-Net expects one file per channel named ``{case_id}_0000.nii.gz``
    (``0000`` = channel index for single-modality CBCT). Accepts either a
    ``.nii`` or a ``.nii.gz`` upload and always produces gzip output.
    """
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    # Clear any previous single-case run so nnU-Net only sees this scan.
    for leftover in input_dir.iterdir():
        leftover.unlink()

    dst = input_dir / f"{case_id}_0000.nii.gz"
    src = str(input_path)
    if src.endswith(".nii.gz"):
        shutil.copy2(src, dst)
    elif src.endswith(".nii"):
        with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        # Unknown extension: try to load & re-save via nibabel.
        img = nib.load(src)
        nib.save(img, str(dst))
    return str(dst)


# --------------------------------------------------------------------------- #
#  Stage 1 — segmentation
# --------------------------------------------------------------------------- #



def _restore_segmentation_to_input_grid(
    seg_path: str | Path,
    input_path: str | Path,
    out_path: str | Path,
    log: Callable[[str], None] = print,
) -> str:
    """Create a COMPLETE multi-label segmentation on the uploaded CBCT grid.

    The returned file always has the original scan's 3-D shape and affine.
    Label values are preserved with nearest-neighbour interpolation.
    """
    seg_img = nib.load(str(seg_path))
    ref_img = nib.load(str(input_path))
    ref_shape = tuple(int(x) for x in ref_img.shape[:3])
    seg_shape = tuple(int(x) for x in seg_img.shape[:3])
    same_shape = seg_shape == ref_shape
    same_affine = np.allclose(seg_img.affine, ref_img.affine, rtol=0.0, atol=1e-5)

    # nnU-Net normally exports directly on the input grid. Copying its
    # already-compressed integer labels avoids another full-volume read,
    # cast, gzip compression and label sort (hundreds of millions of voxels).
    full_path = Path(out_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    proxy = seg_img.dataobj
    unscaled = (getattr(proxy, "slope", 1.0) == 1.0
                and getattr(proxy, "inter", 0.0) == 0.0)
    if (len(seg_img.shape) == 3 and len(ref_img.shape) == 3
            and same_shape and np.array_equal(seg_img.affine, ref_img.affine)
            and np.issubdtype(seg_img.get_data_dtype(), np.integer) and unscaled
            and str(seg_path).lower().endswith(".nii.gz")
            and str(full_path).lower().endswith(".nii.gz")):
        if Path(seg_path).resolve() != full_path.resolve():
            shutil.copyfile(seg_path, full_path)
        log(f"FULL segmentation ready -> {full_path.name} | shape={ref_shape} "
            "| original-grid labels reused without recompression")
        return str(full_path)

    if same_shape and same_affine:
        data = np.rint(np.asanyarray(seg_img.dataobj)).astype(np.int16, copy=False)
    else:
        log(
            "Restoring segmentation to the FULL uploaded CBCT grid "
            f"(prediction={seg_shape}, reference={ref_shape})..."
        )
        restored = nibproc.resample_from_to(
            seg_img, (ref_shape, ref_img.affine), order=0, mode="nearest"
        )
        data = np.rint(np.asanyarray(restored.dataobj)).astype(np.int16, copy=False)

    full_path = Path(out_path)
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_img = nib.Nifti1Image(data, ref_img.affine, ref_img.header.copy())
    full_img.set_data_dtype(np.int16)
    nib.save(full_img, str(full_path))

    log(
        f"FULL segmentation ready -> {full_path.name} | "
        f"shape={ref_shape} | original-grid labels preserved"
    )
    return str(full_path)


def segment_scan(
    input_path: str | Path,
    work_dir: str | Path,
    case_id: str = "scan",
    fold: str = "5",
    device: str = "cuda",
    trainer: Optional[str] = None,
    plans: Optional[str] = None,
    config: Optional[str] = None,
    results_dir: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> str:
    """Run nnU-Net tooth segmentation on ONE scan.

    Parameters
    ----------
    input_path : uploaded CBCT (.nii or .nii.gz).
    work_dir   : scratch directory for this run.
    case_id    : identifier used for intermediate file names.
    fold       : nnU-Net fold to use. The released ToothFairy2 model is
                 run with fold '5' (matches the original batch notebook).
    device     : 'cuda' on the Colab A100, or 'cpu' for a slow fallback.
    trainer/plans/config : auto-detected from the model folder if omitted.

    Returns
    -------
    Path to the predicted segmentation NIfTI ``{case_id}.nii.gz``.
    """
    # Require a GPU only when new segmentation is actually needed. Existing
    # masks can be measured without weights/GPU in folder-reuse mode.
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("This case needs new segmentation, but CUDA is unavailable. "
                               "Select a GPU runtime or explicitly use device='cpu'.")
    results_dir = results_dir or get_results_dir()
    os.environ["nnUNet_results"] = results_dir

    if trainer is None or plans is None or config is None:
        trainer, plans, config = detect_model_config(results_dir)

    work_dir = Path(work_dir)
    input_dir = work_dir / "input"
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    log("Staging scan for the segmentation model...")
    _stage_input(input_path, input_dir, case_id)

    command = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", DATASET_ID,
        "-tr", trainer,
        "-p", plans,
        "-c", config,
        "-f", str(fold),
        "-chk", "checkpoint_final.pth",
        "-device", device,
        "-npp", "1",  # one input case per subprocess: no idle preprocessing pool
        "-nps", "1",  # one export task; retain nnU-Net's resampling and TTA
    ]
    log("Running tooth segmentation (nnU-Net) — this is the slow step...")
    log("$ " + " ".join(command))

    # Stream the model's stdout/stderr line-by-line so the UI can show
    # live progress instead of freezing until completion.
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line:
            log(line)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            f"nnU-Net segmentation failed (exit code {proc.returncode}). "
            f"See the log above."
        )

    seg_path = output_dir / f"{case_id}.nii.gz"
    if not seg_path.exists():
        # nnU-Net occasionally names by the staged stem; find any output.
        candidates = sorted(output_dir.glob("*.nii.gz"))
        if not candidates:
            raise FileNotFoundError(
                f"Segmentation produced no output in {output_dir}."
            )
        seg_path = candidates[0]
    log(f"Segmentation complete -> {seg_path.name}")

    # IMPORTANT: create a full-volume mask on the ORIGINAL uploaded CBCT grid.
    # This exact file is then used for measurement and download.
    full_seg_path = work_dir / f"{case_id}_FULL_MASK.nii.gz"
    return _restore_segmentation_to_input_grid(
        seg_path, input_path, full_seg_path, log=log
    )


# --------------------------------------------------------------------------- #
#  Stage 2 — width measurement
# --------------------------------------------------------------------------- #

def measure_widths(
    seg_path: str | Path,
    arch_maps: Optional[list[tuple[str, dict[int, str], int]]] = None,
    case_id: Optional[str] = None,
    **furcation_kwargs,
) -> "CaseResult":
    """Run the furcation / CR analysis and return arch widths (mm).

    ``**furcation_kwargs`` are forwarded to ``estimate_tooth_cr``
    (``root_count_rule``, ``hull_method``, ``crown_polarity``,
    ``allow_relaxation``), which is how revision-1 behaviour can be
    reproduced for comparison without editing the module.
    """
    arch_maps = arch_maps or DEFAULT_ARCH_MAPS
    return process_case(
        seg_path, case_id=case_id, arch_label_maps=arch_maps,
        **furcation_kwargs
    )


# --------------------------------------------------------------------------- #
#  Full inference path
# --------------------------------------------------------------------------- #

def run_pipeline(
    input_path: str | Path,
    work_dir: str | Path,
    case_id: str = "scan",
    arch_maps: Optional[list[tuple[str, dict[int, str], int]]] = None,
    fold: str = "5",
    device: str = "cuda",
    results_dir: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> tuple[str, "CaseResult"]:
    """Segment one scan, then measure its transverse widths.

    Returns
    -------
    (segmentation_path, CaseResult)
    """
    arch_maps = arch_maps or DEFAULT_ARCH_MAPS
    seg_path = segment_scan(
        input_path, work_dir, case_id=case_id, fold=fold,
        device=device, results_dir=results_dir, log=log,
    )
    log("Measuring transverse basal-bone widths...")
    result = measure_widths(seg_path, arch_maps=arch_maps, case_id=case_id)
    log("Measurement complete.")

    # Report the widths and the Yonsei Transverse Index diagnosis in the log.
    mx = result.arch_widths_mm.get("maxilla")
    md = result.arch_widths_mm.get("mandible")
    if mx is not None:
        sub = classify_arch_width(mx, "maxilla")
        log(f"Maxillary transverse width : {mx:.2f} mm  [{sub['label']}]")
    if md is not None:
        sub = classify_arch_width(md, "mandible")
        log(f"Mandibular transverse width: {md:.2f} mm  [{sub['label']}]")
    if mx is not None and md is not None:
        dx = classify_transverse(mx, md)
        log(f"Yonsei Transverse Index    : {dx['index_mm']:.2f} mm "
            f"(cutoffs {dx['crossbite_cutoff_mm']} / "
            f"{dx['excess_cutoff_mm']} mm)")
        log(f"Diagnosis                  : {dx['label']}")
    else:
        log("Diagnosis                  : not available (an arch is missing).")

    return seg_path, result


# --------------------------------------------------------------------------- #
#  DICOM input handling  (folder / zip of .dcm slices -> one .nii.gz)
# --------------------------------------------------------------------------- #

def _strip_macos_junk(root: str | Path) -> int:
    """Delete AppleDouble (._*) and .DS_Store files that macOS zips carry;
    these are not DICOM and confuse series discovery."""
    removed = 0
    for r, _d, files in os.walk(str(root)):
        for f in files:
            if f.startswith("._") or f == ".DS_Store":
                try:
                    os.remove(os.path.join(r, f)); removed += 1
                except OSError:
                    pass
    return removed


def _dicom_volume_candidates(dicom_root, log=print):
    """Return candidate file lists, largest estimated 3D volume first.

    GDCM supplies the spatial order of slice series. Header dimensions,
    rather than file count, let a single multi-frame volume compete fairly
    with a multi-file series. Standalone .dcm files are also considered
    when GDCM does not list them as a series.
    """
    import SimpleITK as sitk
    root = Path(dicom_root)
    candidates, seen = [], set()

    def add(files):
        files = tuple(str(p) for p in files)
        if not files or files in seen:
            return
        seen.add(files)
        try:
            info = sitk.ImageFileReader()
            info.SetImageIO("GDCMImageIO")
            info.SetFileName(files[0])
            info.ReadImageInformation()
            size = tuple(info.GetSize())
            # Several complete volumes sharing a Series UID must not be
            # stacked as though each were a 2D slice.
            if len(files) > 1 and len(size) >= 3 and size[2] > 1:
                for filename in files:
                    add([filename])
                return
            depth = size[2] if len(size) >= 3 else 1
            voxels = int(size[0]) * int(size[1]) * int(depth) * len(files)
        except Exception as e:
            # Preserve a decoding attempt for an otherwise unlisted file;
            # the converter below reports the actual reader error.
            voxels = 0
            log(f"DICOM header probe failed for {files[0]}: {e}")
        nbytes = sum(Path(p).stat().st_size for p in files)
        candidates.append((voxels, nbytes, files))

    if root.is_file():
        add([root])
    else:
        for directory, dirs, names in os.walk(str(root)):
            dirs[:] = sorted(d for d in dirs
                             if not d.startswith(".") and d != "__MACOSX")
            grouped = set()
            try:
                series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(directory) or ()
            except Exception:
                series_ids = ()
            for sid in sorted(series_ids):
                try:
                    files = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(directory, sid)
                except Exception as e:
                    log(f"Cannot list DICOM series {sid}: {e}")
                    continue
                # Do not sort files: GDCM orders slices using DICOM geometry.
                add(files)
                grouped.update(str(Path(p).resolve()) for p in files)
            for name in sorted(names):
                if name.startswith(".") or not name.lower().endswith(".dcm"):
                    continue
                path = Path(directory) / name
                if str(path.resolve()) not in grouped:
                    add([path])

    candidates.sort(key=lambda c: (-c[0], -c[1], c[2]))
    return [files for _voxels, _nbytes, files in candidates]


def dicom_folder_to_nifti(dicom_root: str | Path,
                          out_path: str | Path,
                          log: Callable[[str], None] = print) -> str:
    """Read a single multi-frame DICOM or a series of DICOM slice files.

    Accepts a file or a recursively searched folder. Geometry comes from
    the DICOM reader and is retained when writing NIfTI. If several image
    candidates exist, try the largest estimated volume first. A single
    2D slice is reported as incomplete input, not expanded into fake 3D.
    """
    import SimpleITK as sitk
    candidates = _dicom_volume_candidates(dicom_root, log=log)
    if not candidates:
        raise RuntimeError(
            "No DICOM images found. Supply one multi-frame .dcm volume "
            "or a folder/ZIP containing a DICOM slice series.")

    errors = []
    img = None
    for files in candidates:
        try:
            if len(files) == 1:
                log(f"Reading single-file DICOM volume: {Path(files[0]).name}")
                reader = sitk.ImageFileReader()
                reader.SetImageIO("GDCMImageIO")
                reader.SetFileName(files[0])
            else:
                log(f"Reading DICOM slice series: {len(files)} files.")
                reader = sitk.ImageSeriesReader()
                reader.SetFileNames(files)
            candidate = reader.Execute()
            # Some readers expose a redundant singleton time dimension.
            # SimpleITK slicing retains the 3D physical geometry.
            if candidate.GetDimension() == 4 and candidate.GetSize()[3] == 1:
                candidate = candidate[:, :, :, 0]
            if candidate.GetNumberOfComponentsPerPixel() != 1:
                raise ValueError("Expected scalar CBCT intensities, not a colour/vector image.")
            if candidate.GetDimension() != 3:
                raise ValueError(
                    f"Expected a 3D CBCT volume, got {candidate.GetDimension()}D. "
                    "A lone 2D slice is insufficient; export the complete volume.")
            if min(candidate.GetSize()) <= 1:
                raise ValueError(
                    f"Image has only one slice along an axis: {candidate.GetSize()}. "
                    "Supply a multi-frame volume or all slices of the scan.")
            spacing = np.asarray(candidate.GetSpacing(), dtype=float)
            if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
                raise ValueError(f"Invalid DICOM voxel spacing: {candidate.GetSpacing()}")
            img = candidate
            break
        except Exception as e:
            detail = f"{Path(files[0]).name} ({len(files)} file(s)): {e}"
            errors.append(detail)
            log(f"DICOM candidate rejected: {detail}")

    if img is None:
        raise RuntimeError("No usable 3D DICOM volume could be read. "
                           + " | ".join(errors[:3]))
    sz, sp = img.GetSize(), img.GetSpacing()
    log(f"Volume {sz[0]}x{sz[1]}x{sz[2]}  spacing "
        f"({sp[0]:.3f}, {sp[1]:.3f}, {sp[2]:.3f}) mm.")
    if img.GetPixelID() not in (sitk.sitkInt16, sitk.sitkInt32, sitk.sitkUInt16):
        img = sitk.Cast(img, sitk.sitkInt16)
    sitk.WriteImage(img, str(out_path))
    log(f"Converted DICOM -> {os.path.basename(str(out_path))}.")
    return str(out_path)


def _case_id_from_name(name: str) -> str:
    cid = name
    for suf in (".nii.gz", ".nii", ".zip", ".dcm"):
        if cid.lower().endswith(suf):
            cid = cid[: -len(suf)]
            break
    return "".join(c if c.isalnum() else "_" for c in cid) or "scan"


def prepare_input_nifti(uploaded_files, work_dir: str | Path,
                        log: Callable[[str], None] = print) -> tuple[str, str]:
    """Turn whatever the user uploaded into a single ``.nii.gz`` on disk.

    Accepts, in order of preference:
      * one ``.nii`` / ``.nii.gz`` volume  -> used directly;
      * one (or more) ``.zip`` of a DICOM folder -> extracted & converted;
      * one multi-frame ``.dcm`` volume, or many ``.dcm`` slice files.

    Returns ``(nifti_path, case_id)``.
    """
    import zipfile
    work_dir = Path(work_dir)
    files = list(uploaded_files)
    names = [f.name for f in files]

    # --- single NIfTI ------------------------------------------------------
    if len(files) == 1 and names[0].lower().endswith((".nii", ".nii.gz")):
        ext = ".nii.gz" if names[0].lower().endswith(".nii.gz") else ".nii"
        p = work_dir / f"upload{ext}"
        p.write_bytes(files[0].getbuffer())
        log(f"Using NIfTI upload: {names[0]}")
        return str(p), _case_id_from_name(names[0])

    # --- otherwise build a DICOM directory --------------------------------
    dicom_dir = work_dir / "dicom_in"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    zips = [f for f in files if f.name.lower().endswith(".zip")]
    if zips:
        for z in zips:
            zpath = work_dir / z.name
            zpath.write_bytes(z.getbuffer())
            log(f"Extracting {z.name} …")
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(dicom_dir)
        case_id = _case_id_from_name(zips[0].name)
    else:
        log(f"Saving {len(files)} uploaded DICOM file(s) …")
        for f in files:
            (dicom_dir / os.path.basename(f.name)).write_bytes(f.getbuffer())
        case_id = _case_id_from_name(names[0]) if len(files) == 1 else "dicom_scan"

    out = work_dir / "converted.nii.gz"
    log("Converting DICOM series to NIfTI …")
    dicom_folder_to_nifti(dicom_dir, out, log=log)
    return str(out), case_id


# --------------------------------------------------------------------------- #
#  Optional visualisation
# --------------------------------------------------------------------------- #

def render_measurement_figure(
    orig_path: str | Path,
    seg_path: str | Path,
    result: "CaseResult",
    arch_maps: Optional[list[tuple[str, dict[int, str], int]]] = None,
    out_png: str | Path = "measurement.png",
) -> Optional[str]:
    """Best-effort figure: for each measurable arch, an axial slice at the
    furcation level with the two first molars highlighted and their CR
    points joined by the measured line.

    Returns the PNG path, or ``None`` if a figure could not be produced.
    Any failure here is swallowed — visualisation must never break the
    numeric result.
    """
    arch_maps = arch_maps or DEFAULT_ARCH_MAPS
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        seg_img = nib.load(str(seg_path))
        seg = np.asanyarray(seg_img.dataobj)
        seg = np.squeeze(seg).astype(np.int32)
        affine = seg_img.affine
        inv_affine = np.linalg.inv(affine)

        orig_img = nib.load(str(orig_path))
        orig = np.squeeze(np.asanyarray(orig_img.dataobj)).astype(np.float32)
        # Normalise CT for display.
        p1, p99 = np.percentile(orig, [1, 99])
        orig = np.clip((orig - p1) / (p99 - p1 + 1e-8), 0, 1)

        # Which arches actually produced a width?
        measurable = [
            (name, lm) for name, lm, *_ in arch_maps
            if result.arch_widths_mm.get(name) is not None
        ]
        if not measurable:
            return None

        fig, axes = plt.subplots(
            1, len(measurable), figsize=(7 * len(measurable), 7)
        )
        if len(measurable) == 1:
            axes = [axes]
        fig.patch.set_facecolor("#0d0d0d")

        for ax, (name, label_map) in zip(axes, measurable):
            labels = list(label_map.keys())
            teeth = [result.per_tooth[l] for l in labels]
            crs_mm = [t.furcation_mm for t in teeth]
            crs_vox = [nib.affines.apply_affine(inv_affine, c) for c in crs_mm]

            # Axial slice (axis 2) at the mean furcation z of the two teeth.
            z = int(round(np.mean([c[2] for c in crs_vox])))
            z = int(np.clip(z, 0, seg.shape[2] - 1))

            ct_slice = orig[:, :, z].T
            ax.imshow(ct_slice, cmap="gray", origin="lower")

            # Overlay just this arch's two teeth.
            seg_slice = seg[:, :, z].T
            overlay = np.zeros((*seg_slice.shape, 4), np.float32)
            colours = [(0.20, 0.85, 1.0), (1.0, 0.55, 0.20)]
            for lbl, col in zip(labels, colours):
                overlay[seg_slice == lbl] = (*col, 0.45)
            ax.imshow(overlay, origin="lower")

            # CR points + connecting measurement line (in-plane x,y).
            xs = [c[0] for c in crs_vox]
            ys = [c[1] for c in crs_vox]
            ax.plot(xs, ys, "-", color="#ffd400", lw=2.0, zorder=5)
            ax.scatter(xs, ys, s=70, c="#ffd400",
                       edgecolors="black", zorder=6)
            for (x, y), lbl in zip(zip(xs, ys), labels):
                ax.text(x, y + 6, label_map[lbl], color="white",
                        fontsize=10, ha="center", zorder=7)

            width = result.arch_widths_mm[name]
            ax.set_title(
                f"{name.capitalize()}  —  {width:.2f} mm",
                color="white", fontsize=15,
            )
            ax.axis("off")

        fig.tight_layout()
        fig.savefig(str(out_png), dpi=130,
                    facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        return str(out_png)
    except Exception:  # noqa: BLE001 — visualisation is non-critical
        return None


# ==========================================================================
# SECTION 3 — Streamlit user interface
# ==========================================================================
"""
app.py
======

Streamlit interface for the CBCT transverse basal-bone width pipeline.

Flow:
    1. User uploads ONE CBCT scan (.nii / .nii.gz) from their computer.
    2. Stage 1 — nnU-Net segments all teeth.
    3. Stage 2 — molar_cr locates the first-molar centres of resistance and
       reports the maxillary and mandibular transverse widths in mm.

No ground truth is required or used. Designed to run on a Google Colab
A100 GPU, exposed through a tunnel (see the Colab launcher notebook).
"""


import os
import tempfile
import time
from pathlib import Path

import streamlit as st
import pandas as pd



# --------------------------------------------------------------------------- #
#  Ground-truth landmark validation — helpers (pure, no Streamlit)
# --------------------------------------------------------------------------- #
# OEM landmark name -> universal tooth code. The code is then resolved to
# whatever integer label the active arch map uses (ToothFairy 6/14/30/22 or FDI
# 16/26/46/36), so this works regardless of which label scheme is selected.
_OEM_TOOTH_CODE = {
    "Maxillary Right": "UR6",
    "Maxillary Left": "UL6",
    "Mandibular Right": "LR6",
    "Mandibular Left": "LL6",
}


def _parse_oem_landmarks(text: str):
    """Parse an OEM 'Landmark' CSV export.

    Returns ({oem_name: xyz array}, coordinate_system_string). Tolerant of the
    export's header rows, CRLF line endings, and the malformed ``"X`` cell.
    """
    pts, coord_system = {}, "?"
    for raw in text.splitlines():
        line = raw.strip().strip("\r")
        if line.lower().startswith('"coordinate system"'):
            parts = line.split(",", 1)
            if len(parts) > 1:
                coord_system = parts[1].strip().strip('"')
        for name in _OEM_TOOTH_CODE:
            if line.startswith('"' + name):
                nums = line.split(",")[1:]
                try:
                    pts[name] = np.array([float(x) for x in nums[:3]], dtype=float)
                except ValueError:
                    pass
    return pts, coord_system


def _oem_name_to_label(arch_maps):
    """Map each OEM landmark name -> integer label via the active arch maps."""
    code_to_label = {}
    for _arch, label_map, *_ in arch_maps:
        for lbl, tooth in label_map.items():
            code_to_label[tooth] = lbl
    return {name: code_to_label[code]
            for name, code in _OEM_TOOTH_CODE.items()
            if code in code_to_label}


def _label_to_tooth(arch_maps):
    out = {}
    for _arch, label_map, *_ in arch_maps:
        for lbl, tooth in label_map.items():
            out[lbl] = tooth
    return out


def _gt_landing_count(seg_data, affine, gt_by_label):
    """How many GT points land inside the volume on their own tooth label."""
    inv = np.linalg.inv(affine)
    shape = np.asarray(seg_data.shape)
    n_hit = 0
    for lbl, w in gt_by_label.items():
        v = inv[:3, :3] @ np.asarray(w, float) + inv[:3, 3]
        vi = np.round(v).astype(int)
        if np.all((vi >= 0) & (vi < shape)) and int(seg_data[tuple(vi)]) == lbl:
            n_hit += 1
    return n_hit


def _gt_landing_distances(seg_data, affine, gt_by_label, tol_vox=8.0):
    """Distance (voxels) from each mapped GT point to the nearest voxel of
    its own tooth label; a point 'lands' when that distance is within
    ``tol_vox``.

    An exact-voxel test is too brittle for furcation landmarks: the centre
    of resistance sits in the notch BETWEEN the separated roots, where the
    segmentation mask itself can be background, and a one-voxel erosion
    shifts the boundary.  Eight voxels (~2.4 mm at 0.3 mm spacing) bridges
    the inter-root notch while remaining far too tight for a mirrored --
    contralateral -- assignment (~130 voxels away) or a wrong grid to
    pass, so the frame/label arbitration is unaffected.  The per-point
    distances are returned so the interface can report them.
    """
    inv = np.linalg.inv(affine)
    shape = np.asarray(seg_data.shape)
    r = int(np.ceil(tol_vox))
    dists = {}
    for lbl, w in gt_by_label.items():
        v = inv[:3, :3] @ np.asarray(w, float) + inv[:3, 3]
        vi = np.round(v).astype(int)
        i0, i1 = max(vi[0] - r, 0), min(vi[0] + r + 1, shape[0])
        j0, j1 = max(vi[1] - r, 0), min(vi[1] + r + 1, shape[1])
        k0, k1 = max(vi[2] - r, 0), min(vi[2] + r + 1, shape[2])
        if i0 >= i1 or j0 >= j1 or k0 >= k1:
            dists[lbl] = None
            continue
        hits = np.argwhere(seg_data[i0:i1, j0:j1, k0:k1] == lbl)
        if len(hits) == 0:
            dists[lbl] = None
            continue
        ctr = np.array([vi[0] - i0, vi[1] - j0, vi[2] - k0], float)
        dists[lbl] = float(np.sqrt(((hits - ctr) ** 2).sum(axis=1)).min())
    n_hit = sum(1 for d in dists.values()
                if d is not None and d <= tol_vox)
    return n_hit, dists


def _swap_lr_labels(arch_maps):
    """Return a copy of ``arch_maps`` with every left/right tooth name
    mirrored (R <-> L), labels unchanged.

    Used to test the mirrored label assignment: the DICOM -> NIfTI
    conversion flips the array handedness the pretrained model was
    trained on, so a scan converted by a different chain can arrive with
    the opposite left/right labelling. The ground-truth landing check
    arbitrates which assignment matches the physical anatomy.
    """
    out = []
    for name, label_map, *rest in arch_maps:
        swapped = {}
        for lbl, tooth in label_map.items():
            t = str(tooth)
            if len(t) >= 3 and t[0] in "UL" and t[1] in "RL":
                t = t[0] + ("L" if t[1] == "R" else "R") + t[2:]
            swapped[lbl] = t
        out.append((name, swapped, *rest))
    return out


def _gt_zero_param_map(gt_by_label, affine, seg_shape, source):
    """Map OEM ground-truth points into the scan's world frame with ZERO
    fitted parameters.

    The landmarks were digitised on the same physical grid the scan's
    voxels occupy, so:

    ``source="image"``   -- OEM Image CS mm -> voxel index = xyz / spacing
    ``source="patient"`` -- OEM Patient CS mm -> Image CS mm first: the
        OEM centres that frame at the volume centre, an offset computed
        from the scan's own shape and spacing (not fitted to the
        landmarks); then divide by the spacing as above.

    The voxel index is taken through the scan's NIfTI affine to world
    millimetres. Assumes an axis-aligned (orthogonal) grid -- which the
    landing check verifies per case: if the assumption fails, the mapped
    points do not land on their teeth and the caller falls back to the
    frame-recovery cascade.
    """
    A = np.asarray(affine, float)
    spacing = np.linalg.norm(A[:3, :3], axis=0)
    centre = (np.asarray(seg_shape, float) - 1.0) * spacing / 2.0
    out = {}
    for lbl, xyz in gt_by_label.items():
        p = np.asarray(xyz, float)
        if np.isnan(p).any():
            continue
        img = p if source == "image" else p + centre
        vox = img / spacing
        out[lbl] = A[:3, :3] @ vox + A[:3, 3]
    return out


def _pairwise_shape_report(pred_by_label, gt_by_label):
    """Frame-invariant shape comparison: every pairwise distance between
    landmarks, in both point sets.

    Rotations and translations leave ALL pairwise distances unchanged, so
    any mismatch between the two sets' distance tables is genuine shape
    (localization) disagreement -- never a coordinate-frame effect. This is
    the arbiter that separates a true frame rotation from real per-landmark
    error when the raw offsets are not constant.
    """
    labels = sorted(set(pred_by_label) & set(gt_by_label))
    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            pa = np.asarray(pred_by_label[a], float)
            pb = np.asarray(pred_by_label[b], float)
            ga = np.asarray(gt_by_label[a], float)
            gb = np.asarray(gt_by_label[b], float)
            if (np.isnan(pa).any() or np.isnan(pb).any()
                    or np.isnan(ga).any() or np.isnan(gb).any()):
                continue
            dp = float(np.linalg.norm(pa - pb))
            dg = float(np.linalg.norm(ga - gb))
            rows.append({"a": a, "b": b, "pred_mm": dp, "gt_mm": dg,
                         "err_mm": dp - dg})
    return rows


def _fit_rigid_transform(src_pts, dst_pts):
    """Least-squares similarity fit (Kabsch) mapping src_pts -> dst_pts.

    Returns rotation angle (deg), isotropic scale, translation, and
    per-point residuals (mm). Used ONLY to characterise the relationship
    between the OEM's own Image CS and Patient CS exports from the shared
    landmarks -- never to align ground truth to predictions before scoring.
    """
    P = np.asarray(src_pts, float)
    Q = np.asarray(dst_pts, float)
    Pc, Qc = P.mean(axis=0), Q.mean(axis=0)
    X, Y = P - Pc, Q - Qc
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    d = float(np.sign(np.linalg.det(Vt.T @ U.T)))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    scale = float(S.sum() / float((X ** 2).sum()))
    t = Qc - scale * (Pc @ R.T)
    fitted = scale * (P @ R.T) + t
    resid = np.linalg.norm(fitted - Q, axis=1)
    ang = float(np.degrees(np.arccos(
        np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
    return {"R": R, "t": t, "scale": scale, "angle_deg": ang,
            "residuals_mm": resid,
            "rms_mm": float(np.sqrt((resid ** 2).mean()))}


def _signed_permutation_mats():
    """All 48 signed permutation matrices -- every axis convention one
    software frame can differ from another by (axis-order swaps and
    LPS/RAS-style flips)."""
    import itertools
    mats = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3))
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for signs in itertools.product((1.0, -1.0), repeat=3):
            mats.append(np.diag(signs) @ P)
    return mats


def _describe_axis_map(M):
    """Human-readable axis convention, e.g. 'GT +x -> scan -y, ...'."""
    axes = "xyz"
    parts = []
    for j in range(3):
        col = M[:, j]
        i = int(np.argmax(np.abs(col)))
        s = "+" if col[i] > 0 else "-"
        parts.append(f"GT +{axes[j]} \u2192 scan {s}{axes[i]}")
    return ", ".join(parts)


def _tooth_cr_proxy(seg_data, label, affine):
    """Prediction-free furcation-level anchor for a tooth: the centroid of
    the mid-band of the tooth along its crown->root axis, measured from the
    crown end. The whole-blob centroid is pulled apically by the root mass;
    the mid-band centroid sits much closer to the furcation centre, which
    shrinks the systematic anchor offset in frame recovery. The band spans
    20-50% of the tooth length from the crown tip, centred on ~35% -- the
    anatomical furcation level of a first molar (crown ~7 mm, roots
    ~12 mm).

    The crown->root axis is NOT simply the first principal direction: a
    molar's crown is wide and its roots splay, so raw PCA variance can be
    dominated by the crown's width. Instead, all three principal directions
    are tried and the one whose two ENDS differ most in width is chosen --
    the wide end is the crown, the narrow end the root apices. Falls back
    to the blob centroid when no direction discriminates."""
    idx = np.argwhere(seg_data == label)
    if len(idx) < 50:
        return None
    A = np.asarray(affine, float)[:3, :3]
    origin = np.asarray(affine, float)[:3, 3]
    world = idx @ A.T + origin
    c = world.mean(axis=0)
    X = world - c
    try:
        _, _, vt = np.linalg.svd(X, full_matrices=False)
    except Exception:  # noqa: BLE001
        return c
    best = None
    for axis in vt:
        t = X @ axis
        lo, hi = float(t.min()), float(t.max())
        if hi - lo < 1.0:
            continue
        rad = np.linalg.norm(X - np.outer(t, axis), axis=1)
        end_lo = float(rad[t < lo + 0.1 * (hi - lo)].mean())
        end_hi = float(rad[t > hi - 0.1 * (hi - lo)].mean())
        asym = abs(end_hi - end_lo)
        if best is None or asym > best[0]:
            best = (asym, t, lo, hi, end_lo, end_hi)
    if best is None:
        return c
    _, t, lo, hi, end_lo, end_hi = best
    if end_hi > end_lo:
        lo, hi = -hi, -lo
        t = -t
    band = (t >= lo + 0.20 * (hi - lo)) & (t <= lo + 0.50 * (hi - lo))
    if int(band.sum()) < 20:
        return c
    return world[band].mean(axis=0)


def _recovery_tolerance(recov, gt_by_label):
    """Case-specific frame-recovery tolerance (mm), computed from the fit's
    own diagnostics: residual rotation and scale deviation act on the
    landmark configuration radius, and the anchoring rms adds the
    point-placement noise. Per-landmark differences below this value are
    frame-recovery artefact, not measured error."""
    G = np.array([np.asarray(gt_by_label[l], float) for l in recov["labels"]])
    R = float(np.sqrt(((G - G.mean(axis=0)) ** 2).sum(axis=1).mean()))
    rot = abs(float(np.sin(np.radians(recov["angle_deg"])))) * R
    scl = abs(1.0 - recov["scale"]) * R
    anc = float(recov["rms_mm"])
    return float(np.sqrt(rot ** 2 + scl ** 2 + anc ** 2))


def _lin_ccc(x, y):
    """Lin's concordance correlation coefficient between two measurers."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    cov = float(np.cov(x, y, ddof=1)[0, 1])
    return float(2.0 * cov / (vx + vy + (x.mean() - y.mean()) ** 2))


def _icc_2_1(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single
    measurement (Shrout & Fleiss). x and y are the two raters over n
    subjects. Returns None when n < 3 or the denominator degenerates."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    if n < 3:
        return None
    data = np.column_stack([x, y])
    k = 2
    mean_r = data.mean(axis=1)
    mean_c = data.mean(axis=0)
    grand = data.mean()
    ss_r = k * ((mean_r - grand) ** 2).sum()
    ss_c = n * ((mean_c - grand) ** 2).sum()
    ss_e = ((data - mean_r[:, None] - mean_c[None, :] + grand) ** 2).sum()
    ms_r = ss_r / (n - 1)
    ms_c = ss_c / (k - 1)
    ms_e = ss_e / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    if denom <= 0:
        return None
    return float((ms_r - ms_e) / denom)


def _recover_frame_via_teeth(gt_by_label, seg_data, affine,
                             molar_labels=(6, 14, 22, 30),
                             tol_rms_mm=1.0, tol_point_mm=1.5,
                             margin=2.0):
    """Recover the GT -> scan frame transform using the segmentation as the
    anchor -- never the predicted CRs.

    Each GT landmark claims to sit on a specific molar, and the scan's own
    segmentation says where that molar is. A software frame can only differ
    by an axis convention, so all 48 signed permutations are searched. For
    each, a single-shot similarity fit (rotation + scale + translation)
    maps the GT points onto the tooth centroids; the candidate is then
    scored by how far the mapped points lie from their own tooth regions.
    The correct convention lands every point INSIDE its tooth (~0 mm);
    wrong conventions leave points millimetres outside. The winner must be
    decisive (clearly better than the runner-up) or the recovery is
    rejected. Because the predicted CRs are never used, the recovered
    frame cannot absorb the localization error it is later used to
    measure. Returns None when no convention anchors convincingly.
    """
    from scipy.spatial import cKDTree

    if seg_data is None or affine is None:
        return None
    labels = []
    for l in molar_labels:
        g = gt_by_label.get(l)
        if g is None:
            continue
        g = np.asarray(g, float)
        if g.shape != (3,) or np.isnan(g).any():
            continue
        if int((seg_data == l).sum()) > 50:
            labels.append(l)
    if len(labels) < 3:
        return None

    A = np.asarray(affine, float)[:3, :3]
    origin = np.asarray(affine, float)[:3, 3]
    trees, centroids = {}, {}
    for l in labels:
        idx = np.argwhere(seg_data == l)
        if len(idx) > 30000:
            idx = idx[:: int(np.ceil(len(idx) / 30000.0))]
        world = idx @ A.T + origin
        trees[l] = cKDTree(world)
        proxy = _tooth_cr_proxy(seg_data, l, affine)
        centroids[l] = proxy if proxy is not None else world.mean(axis=0)

    def _eval_subset(Gs, sub_labels):
        Cs = np.array([centroids[l] for l in sub_labels], float)
        cands = []
        for M in _signed_permutation_mats():
            fit = _fit_rigid_transform(Gs @ M.T, Cs)
            Rm, s, t = fit["R"] @ M, fit["scale"], fit["t"]
            Pw = s * (Gs @ Rm.T) + t
            d = np.array([trees[l].query(p)[0]
                          for l, p in zip(sub_labels, Pw)])
            rms = float(np.sqrt((d ** 2).mean()))
            cands.append(dict(M=M, R=Rm, scale=s, t=t,
                              per_point_mm=d, rms_mm=rms,
                              cen_rms_mm=fit["rms_mm"]))
        cands.sort(key=lambda c: c["rms_mm"])
        best = cands[0]
        # Runner-up = best candidate that maps the landmarks to genuinely
        # DIFFERENT positions. Many permutations converge to the same
        # mapping of the four points (the fit absorbs the difference, and
        # near-coplanar configs add a harmless out-of-plane reflection
        # twin); those are the same solution, not an ambiguity. Only a
        # near-tie between different mappings makes recovery ambiguous.
        Pb = best["scale"] * (Gs @ best["R"].T) + best["t"]
        runner = np.inf
        twins = [best]
        for c in cands[1:]:
            Pc = c["scale"] * (Gs @ c["R"].T) + c["t"]
            if float(np.linalg.norm(Pc - Pb, axis=1).mean()) > 2.0:
                runner = c["rms_mm"]
                break
            twins.append(c)
        # Among equivalent mappings report the one closest to a pure axis
        # convention (smallest residual rotation) -- cosmetic only.
        def _res_ang(c):
            Rr = c["R"] @ c["M"].T
            return float(np.degrees(np.arccos(np.clip(
                (np.trace(Rr) - 1.0) / 2.0, -1.0, 1.0))))
        best = min(twins, key=_res_ang)
        return best, runner

    def _passes(b, r):
        return (b is not None
                and b["rms_mm"] <= tol_rms_mm
                and float(b["per_point_mm"].max()) <= tol_point_mm
                and b["rms_mm"] * margin <= r)

    G = np.array([gt_by_label[l] for l in labels], float)
    best, runner = _eval_subset(G, labels)
    used, outlier = list(labels), None

    if len(labels) >= 4:
        # Leave-one-out: a single GT point may be off its tooth entirely
        # (mis-annotation). Refit on each 3-subset when the full fit fails,
        # or when dropping one point improves the fit dramatically (the
        # 4-point fit otherwise spreads a moderate error across all teeth
        # and hides it). Flag the excluded landmark instead of failing.
        full_ok = _passes(best, runner)
        better = None
        for drop in labels:
            keep = [l for l in labels if l != drop]
            idx = [labels.index(l) for l in keep]
            b2, r2 = _eval_subset(G[idx], keep)
            if not _passes(b2, r2):
                continue
            if (not full_ok) or (best["cen_rms_mm"] > 1.0
                                 and b2["cen_rms_mm"]
                                 < 0.6 * best["cen_rms_mm"]):
                if better is None or b2["cen_rms_mm"] < better[0]["cen_rms_mm"]:
                    better = (b2, r2, keep, drop)
        if better is not None:
            best, runner, used, outlier = better
    if not _passes(best, runner):
        return None

    def _apply(g):
        return best["scale"] * (np.asarray(g, float) @ best["R"].T) + best["t"]

    R_res = best["R"] @ best["M"].T
    ang = float(np.degrees(np.arccos(
        np.clip((np.trace(R_res) - 1.0) / 2.0, -1.0, 1.0))))
    out = dict(labels=used,
               axis_map=_describe_axis_map(best["M"]),
               angle_deg=ang,
               scale=float(best["scale"]),
               t=best["t"],
               rms_mm=best["rms_mm"],
               per_point={l: float(d)
                          for l, d in zip(used, best["per_point_mm"])},
               outlier=outlier,
               apply=_apply)
    if outlier is not None:
        out["outlier_dist_mm"] = float(
            trees[outlier].query(_apply(gt_by_label[outlier]))[0])
    return out


def _reconcile_frame(pred_by_label, gt_by_label, seg_data, affine,
                     const_tol_mm: float = 1.5, shape_tol_mm: float = 2.0):
    """Reconcile the OEM frame to the scan frame without a per-case 6-DoF fit.

    Modes: 'identity' | 'translation' | 'needs-transform' | 'insufficient'.
    A pure translation only removes the frame-origin difference (verified by the
    offset being constant across landmarks); it never fits a rotation to the
    points being scored.
    """
    labels = sorted(set(pred_by_label) & set(gt_by_label))
    P = np.array([pred_by_label[l] for l in labels], dtype=float) if labels \
        else np.empty((0, 3))
    G = np.array([gt_by_label[l] for l in labels], dtype=float) if labels \
        else np.empty((0, 3))
    if len(labels):
        valid = ~(np.isnan(P).any(axis=1) | np.isnan(G).any(axis=1))
        labels = [l for l, ok in zip(labels, valid) if ok]
        P, G = P[valid], G[valid]
    if len(labels) < 2:
        return dict(mode="insufficient", labels=labels, gt_aligned=None)

    if seg_data is not None and affine is not None:
        n_hit = _gt_landing_count(seg_data, affine,
                                  {l: g for l, g in zip(labels, G)})
        if n_hit == len(labels):
            return dict(mode="identity", labels=labels,
                        gt_aligned={l: g for l, g in zip(labels, G)})

    offset = P - G
    t = offset.mean(axis=0)
    spread = offset.std(axis=0)
    if float(np.linalg.norm(spread)) <= const_tol_mm:
        Ga = G + t
        return dict(mode="translation", labels=labels,
                    frame_offset=t, offset_spread=spread,
                    gt_aligned={l: g for l, g in zip(labels, Ga)})
    # The offsets are not constant, so the frames differ by more than a
    # translation. Arbitrate the cause before reporting: a true frame
    # ROTATION leaves every pairwise landmark distance unchanged, while
    # genuine localization disagreement changes them. Compare the two
    # distance tables -- frame-invariant, so this test itself is valid
    # regardless of any frame mismatch.
    shape = _pairwise_shape_report(pred_by_label, gt_by_label)
    max_shape_err = max((abs(r["err_mm"]) for r in shape), default=0.0)
    if max_shape_err <= shape_tol_mm:
        # Shapes agree: the mismatch really is a frame rotation.
        return dict(mode="needs-transform", labels=labels,
                    offset_spread=spread, gt_aligned=None, shape=shape)
    # Shapes genuinely differ: this is localization disagreement, not a
    # coordinate-frame problem.
    return dict(mode="shape-mismatch", labels=labels,
                offset_spread=spread, gt_aligned=None, shape=shape)


def _bilateral_width(by_label, right_label, left_label):
    if right_label in by_label and left_label in by_label:
        a = np.asarray(by_label[right_label], float)
        b = np.asarray(by_label[left_label], float)
        if not (np.any(np.isnan(a)) or np.any(np.isnan(b))):
            return float(np.linalg.norm(a - b))
    return None


# --------------------------------------------------------------------------- #
#  Batch mode — a folder of scans -> cohort CSVs (pure, no Streamlit)
# --------------------------------------------------------------------------- #

import re

def _best_dcm_series_dir(pkg):
    """Find a DICOM-containing directory inside an Invivo package.

    Accept one multi-frame .dcm or multiple slice files. Payload size is a
    discovery hint only; conversion examines all package series and ranks
    their estimated volume size using DICOM headers.
    """
    best, best_bytes = None, 0
    for dp, _dn, fn in os.walk(str(pkg)):
        dcm = [f for f in fn if f.lower().endswith(".dcm") and not f.startswith(".")]
        if not dcm:
            continue
        nbytes = sum(os.path.getsize(os.path.join(dp, f)) for f in dcm)
        if nbytes > best_bytes:
            best, best_bytes = Path(dp), nbytes
    return best


def _find_config_inv(pkg):
    """Locate the native volume file inside an Invivo package: prefer a
    ``*Config*.inv`` file, else the largest ``.inv`` file present."""
    cands = [f for f in Path(pkg).rglob("*.inv") if f.is_file()]
    if not cands:
        return None
    cfg = [f for f in cands if "config" in f.name.lower()]
    return cfg[0] if cfg else max(cands, key=lambda f: f.stat().st_size)


def _config_inv_to_nifti(cfg_path, out_path, log=print):
    """Best-effort reader for Invivo's native ``*.Config.inv`` volume -- used
    ONLY when a project package holds no usable DICOM series.

    The header (first ~256 KB) is scanned for volume dimensions, voxel
    spacing and origin in any of the common XML-ish spellings; the raw
    voxel payload is taken from the end of the file (uint8 or 16-bit,
    whichever matches the declared voxel count). Dimensions and spacing are
    REQUIRED and never guessed; orientation is not trusted either -- the
    downstream zero-parameter landing check verifies the frame against any
    GT landmarks and withholds validation when it is wrong.
    """
    raw = Path(cfg_path).read_bytes()
    head = raw[:262144].decode("latin-1", errors="replace")

    def _triple(tag):
        m = re.search(
            tag + r'[^<>]{0,200}?x\s*=\s*"?\s*([\d.+-]+)"?[^<>]{0,200}?'
                  r'y\s*=\s*"?\s*([\d.+-]+)"?[^<>]{0,200}?'
                  r'z\s*=\s*"?\s*([\d.+-]+)"?', head, re.I)
        return tuple(float(g) for g in m.groups()) if m else None

    dims = (_triple(r"VolumeDimensions?") or _triple(r"VolumeSize")
            or _triple(r"Dimensions?"))
    sp = (_triple(r"VoxelSpacing") or _triple(r"VoxelSize")
          or _triple(r"Spacing"))
    org = _triple(r"Origin") or (0.0, 0.0, 0.0)
    if dims is None or sp is None:
        raise RuntimeError(
            f"{Path(cfg_path).name}: could not read volume dimensions/"
            "spacing from the native header -- export DICOM from Invivo for "
            "this case instead (File > Export > DICOM).")
    nx, ny, nz = (int(round(v)) for v in dims)
    nvox = nx * ny * nz
    arr = None
    for dt, nb in ((np.uint8, nvox), ("<i2", 2 * nvox), ("<u2", 2 * nvox)):
        if 0 <= len(raw) - nb < 4_000_000:          # header must be small
            arr = np.frombuffer(raw[len(raw) - nb:], dtype=dt)
            break
    if arr is None:
        raise RuntimeError(
            f"{Path(cfg_path).name}: payload size does not match the "
            f"declared volume {nx}x{ny}x{nz} -- unsupported variant; export "
            "DICOM from Invivo for this case instead.")
    vol = arr.reshape((nz, ny, nx)).transpose(2, 1, 0)    # x fastest
    affine = np.diag([sp[0], sp[1], sp[2], 1.0])
    affine[:3, 3] = org
    nib.save(nib.Nifti1Image(vol.astype(np.int16), affine), str(out_path))
    log(f"Config.inv native volume {nx}x{ny}x{nz}, spacing "
        f"({sp[0]:.3f}, {sp[1]:.3f}, {sp[2]:.3f}) mm.")
    return str(out_path)


_GENERIC_DIRS = {"dicom", "dicoms", "dcm", "ct", "cbct", "img", "image", "images",
                 "scan", "scans", "volume", "volumes", "data", "nifti", "nifti_export",
                 "pre", "post", "pre cbct", "post cbct", "pre_cbct", "post_cbct"}


def _batch_mask_path(case):
    """Full labelled mask beside a file scan, or inside a scan folder/package."""
    source = Path(case["path"]).resolve()
    folder = source if case["kind"] in ("dcm_dir", "inv_pkg") else source.parent
    return folder / f"{case['case_id']}_FULL_MASK.nii.gz"


MASK_FALLBACK_ROOT = Path("/content/drive/MyDrive/CBCT_masks")
CSV_FALLBACK_ROOT = Path("/content/drive/MyDrive/CBCT_batch_output")


def _batch_mask_candidates(case):
    """Case-specific current/legacy mask names; never search arbitrary NIfTIs."""
    destination = _batch_mask_path(case)
    folders = (destination.parent, MASK_FALLBACK_ROOT / str(case["case_id"]))
    names = [f"{case['case_id']}{suffix}{extension}"
             for suffix in ("_FULL_MASK", "_seg")
             for extension in (".nii.gz", ".nii")]
    source = Path(case["path"]).resolve()
    return list(dict.fromkeys((folder / name).resolve()
                             for folder in folders for name in names
                             if (folder / name).resolve() != source))


def _find_reusable_batch_mask(case, input_path, work_dir, log=print):
    """Find a readable label mask matching the scan grid, even without a CSV.

    Filenames and geometry identify candidates, not their model provenance.
    Reused legacy masks are explicitly marked as model/fold-unverified.
    The candidate is copied locally before reading; no source file is changed.
    """
    candidates = []
    for path in _batch_mask_candidates(case):
        try:
            if path.is_file() and path.stat().st_size > 0:
                candidates.append(path)
        except OSError as error:
            log(f"Cannot inspect mask {path}: {error}")
    if not candidates:
        return None
    reference = nib.load(str(input_path))
    if len(reference.shape) != 3:
        raise ValueError(f"Expected a 3-D reference scan, got {reference.shape}")
    if (not np.all(np.isfinite(reference.affine))
            or np.linalg.matrix_rank(reference.affine[:3, :3]) != 3):
        raise ValueError("Reference scan has invalid geometry")
    for candidate in candidates:
        local = None
        try:
            if not candidate.is_file() or candidate.stat().st_size <= 0:
                continue
            extension = ".nii.gz" if candidate.name.endswith(".nii.gz") else ".nii"
            local = Path(work_dir) / f"existing_mask{extension}"
            shutil.copyfile(candidate, local)
            img = nib.load(str(local))
            if len(img.shape) != 3 or img.shape != reference.shape:
                raise ValueError(f"mask shape {img.shape} does not match scan {reference.shape}")
            if (not np.all(np.isfinite(img.affine))
                    or not np.allclose(img.affine, reference.affine, rtol=0, atol=1e-5)):
                raise ValueError("mask origin/orientation/spacing does not match scan")
            # A full decode catches truncated gzip/payloads, not just valid headers.
            # Work in small slabs after decoding to avoid large temporary arrays.
            labels = np.asanyarray(img.dataobj)
            if labels.dtype.kind not in "biuf":
                raise ValueError("mask must contain numeric label values")
            has_tooth_label = False
            for start in range(0, labels.shape[2], 16):
                block = labels[:, :, start:start + 16]
                if not np.all(np.isfinite(block)) or np.any(block < 0):
                    raise ValueError("mask has invalid or negative labels")
                if labels.dtype.kind == "f" and not np.all(block == np.rint(block)):
                    raise ValueError("mask contains probabilities/intensities, not integer labels")
                if np.any(block > 32):
                    raise ValueError("expected Dataset121 tooth labels 0..32, not raw scan intensities/another label scheme")
                has_tooth_label |= bool(np.any(block > 1))
            if not has_tooth_label:
                raise ValueError("empty/binary mask is not a full multi-label tooth segmentation")
            del labels
            log(f"Found valid existing mask -> {candidate}")
            log("Reusing existing mask; skipping nnU-Net. Compatible grid/labels checked; patient identity and original model/fold are not verified by geometry alone.")
            return candidate, str(local)
        except Exception as error:
            log(f"Ignoring unusable mask {candidate}: {error}")
            if local is not None:
                local.unlink(missing_ok=True)
    return None


def _atomic_copy_file(source, destination):
    """Publish only a complete copy; leave an existing destination intact on error."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    size = source.stat().st_size
    if size <= 0:
        raise ValueError(f"Source file is empty: {source}")
    if source == destination:
        return size
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial",
        dir=str(destination.parent))
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        if temporary.stat().st_size != size:
            raise OSError(f"Incomplete mask copy to {destination}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return size


def _save_batch_mask(seg_path, case, log=print):
    """Save full labels beside the scan, with the read-only Drive fallback."""
    destination = _batch_mask_path(case)
    if destination == Path(case["path"]).resolve():
        raise ValueError(f"Mask destination would overwrite the scan: {destination}")
    try:
        size = _atomic_copy_file(seg_path, destination)
    except OSError as error:
        destination = MASK_FALLBACK_ROOT / str(case["case_id"]) / destination.name
        log(f"Cannot write beside scan ({error}); trying {destination}")
        size = _atomic_copy_file(seg_path, destination)
    log(f"Saved full segmentation mask -> {destination}")
    return destination, size


def _batch_mask_is_saved(case, previous):
    """Recognise both notebook CSV formats, including read-only Drive fallback."""
    destination = _batch_mask_path(case)
    try:
        # Accept only locations belonging to this case, not arbitrary CSV paths.
        allowed = set(_batch_mask_candidates(case))
        source = previous.get("source_path")
        if isinstance(source, str) and source and source != str(Path(case["path"]).resolve()):
            return False
        recorded = previous.get("mask_path")
        if not isinstance(recorded, str) or not recorded:
            recorded = previous.get("segmentation_path")
        if not isinstance(recorded, str) or not recorded:
            return False
        saved = Path(recorded).resolve()
        if saved not in allowed or not saved.is_file() or saved.stat().st_size <= 0:
            return False
        if previous.get("mask_status") == "saved":
            return saved.stat().st_size == int(previous["mask_size_bytes"])
        # Legacy read-only-fix CSVs recorded the actual copy path but not size.
        return previous.get("segmentation_save_location") in ("beside_scan", "MyDrive_fallback")
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        return False


def _discover_batch_cases(root):
    """Walk ``root`` and classify every scan it finds as a batch case.

    A case is one of:
      * a ``.nii`` / ``.nii.gz`` volume,
      * a ``.zip`` archive (assumed to hold a DICOM series),
      * a directory containing one multi-frame ``.dcm`` or a slice series,
      * an Invivo ``.inv`` project PACKAGE (one case per package; the
        DICOM volume candidates inside are tried, ``*Config.inv`` is the
        fallback, landmark CSVs at the package root attach automatically).

    Directories that ARE a case are not descended into further; the batch
    output folder and OS junk are skipped. A DICOM-folder case takes its id
    from the nearest non-generic ancestor folder (``case001/pre
    CBCT/DICOM`` -> ``case001``). Ground-truth landmark exports
    (``*ImageCS*.csv`` / ``*PatientCS*.csv``, any casing) attach to a case
    when they sit in its folder or an ancestor folder (up to 4 levels,
    never above ``root``) AND either carry the case id (or its leading
    number) in the filename, or are the only candidate in a folder hosting
    just this one case -- an ambiguous CSV is never silently attached.
    """
    root = Path(root).resolve()
    skip_names = {"__MACOSX", ".git", "batch_output"}
    cases = []

    def _ancestors(d, n=4):
        out = [d]
        for _ in range(n):
            if d == root or d.parent == d:
                break
            d = d.parent
            out.append(d)
        return out

    def _dir_case_id(dp):
        p = dp
        while p.name.lower().strip() in _GENERIC_DIRS and p != root and p.parent != p:
            p = p.parent
        return _case_id_from_name(p.name)

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d not in skip_names and not d.startswith(".")]
        dp = Path(dirpath)
        if dp.name.lower().endswith(".inv"):
            # Invivo project package: ONE case per package. The CT series is
            # selected from the package's DICOM candidates (one file or a
            # slice series); Config.inv is fallback; GT attaches at the root.
            cid = _case_id_from_name(dp.stem)
            series = _best_dcm_series_dir(dp)
            cfg = _find_config_inv(dp)
            if series is not None or cfg is not None:
                gdirs = [dp, dp.parent]
                if series is not None:
                    gdirs += [series, series.parent]
                cases.append({"case_id": cid, "kind": "inv_pkg", "path": dp,
                              "series_dir": series, "config_inv": cfg,
                              "gt_dirs": gdirs})
            dirnames[:] = []          # package consumed; do not descend
            continue
        n_dcm = sum(1 for f in filenames
                    if f.lower().endswith(".dcm") and not f.startswith("."))
        if n_dcm >= 1:
            cases.append({"case_id": _dir_case_id(dp), "kind": "dcm_dir",
                          "path": dp, "gt_dirs": _ancestors(dp)})
            dirnames[:] = []          # do not descend into a case folder
            continue
        for f in sorted(filenames):
            fl = f.lower()
            # Our own exports and unfinished copies must never become scans.
            if fl.startswith(".") or fl.endswith((
                    "_full_mask.nii", "_full_mask.nii.gz",
                    "_seg.nii", "_seg.nii.gz")):
                continue
            if fl.endswith((".nii", ".nii.gz")):
                kind = "nifti"
            elif fl.endswith(".zip"):
                kind = "zip"
            else:
                continue
            cases.append({"case_id": _case_id_from_name(f), "kind": kind,
                          "path": dp / f, "gt_dirs": _ancestors(dp)})

    # A directory "hosts" a case when it lies on the case's GT search chain.
    # Unique-fallback attachment is only allowed from a directory hosting
    # exactly one case, so a shared CSV in a multi-case folder never leaks.
    hosts = {}
    for c in cases:
        for d in c["gt_dirs"]:
            hosts[d] = hosts.get(d, 0) + 1

    def _gt_for(c):
        cid = c["case_id"].lower()
        mnum = re.match(r"\d+", cid)
        keys = {cid} | ({mnum.group(0)} if mnum else set())
        def pick(kind):
            per_dir = []
            for d in c["gt_dirs"]:
                cands = [f for f in sorted(d.glob("*.csv"))
                         if kind in f.name.lower() and "cs" in f.name.lower()]
                per_dir.append((d, cands))
            for d, cands in per_dir:                      # explicit id match wins
                hit = [f for f in cands if any(k in f.name.lower() for k in keys)]
                if hit:
                    return hit[0]
            for d, cands in per_dir:                      # safe unique fallback
                if len(cands) == 1 and hosts.get(d, 0) == 1:
                    return cands[0]
            return None
        return pick("image"), pick("patient")

    # A repeated basename (or pre/post study) is still a separate scan.
    # Attach GT using the original ID, then add a stable relative-path suffix
    # only for collisions so every scan gets its own mask and CSV row.
    import hashlib
    counts = {}
    for c in cases:
        counts[c["case_id"]] = counts.get(c["case_id"], 0) + 1
    seen = {}
    for c in cases:
        img, pat = _gt_for(c)
        c.pop("gt_dirs")
        c["gt_image"], c["gt_patient"] = img, pat
        if counts[c["case_id"]] > 1:
            relative = c["path"].relative_to(root).as_posix()
            suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
            c["case_id"] = f"{c['case_id']}__{suffix}"
        if c["case_id"] in seen:
            raise ValueError(f"Duplicate case ID after disambiguation: {c['case_id']}")
        seen[c["case_id"]] = c
    return [seen[k] for k in sorted(seen)]

def _prepare_input_from_path(case, work_dir, log=print):
    """Path-based twin of ``prepare_input_nifti`` for batch mode."""
    import zipfile
    work_dir = Path(work_dir)
    kind, src, cid = case["kind"], Path(case["path"]), case["case_id"]
    if kind == "nifti":
        dst = work_dir / ("input.nii.gz"
                          if src.name.lower().endswith(".nii.gz")
                          else "input.nii")
        shutil.copy2(src, dst)
        return str(dst), cid
    if kind == "inv_pkg":
        series, cfg = case.get("series_dir"), case.get("config_inv")
        if series is not None:
            try:
                out = work_dir / "converted.nii.gz"
                log("Converting package DICOM volume (single file or slice series) ...")
                # Search the whole package: a compressed single-file volume
                # may be smaller on disk than a different scout directory.
                dicom_folder_to_nifti(src, out, log=log)
                return str(out), cid
            except Exception as e:  # noqa: BLE001
                log(f"package DICOM series failed ({e}); "
                    "trying the native Config.inv volume ...")
        if cfg is not None:
            out = work_dir / "converted.nii.gz"
            _config_inv_to_nifti(cfg, out, log=log)
            return str(out), cid
        raise RuntimeError("Invivo package holds neither a usable DICOM "
                           "series nor a Config.inv volume.")
    if kind == "zip":
        dicom_dir = work_dir / "dicom_in"
        dicom_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(dicom_dir)
    else:                                    # dcm_dir
        dicom_dir = src
    out = work_dir / "converted.nii.gz"
    log("Converting DICOM volume (single multi-frame or slice series) to NIfTI ...")
    dicom_folder_to_nifti(dicom_dir, out, log=log)
    return str(out), cid


def _zero_param_validate(pred_by_label, gt_image_pts, gt_patient_pts,
                         arch_maps, seg_data, affine):
    """Headless ground-truth validation for batch mode.

    Mirrors the interactive zero-parameter path: map the OEM landmark
    exports onto the scan's voxel grid with no fitted parameters (Image CS
    by division by the voxel spacing; Patient CS after re-centring to the
    volume centre, an offset computed from the scan's own shape and
    spacing), verify every mapped point lands on its own tooth, and test
    BOTH left/right label assignments, switching when the mirrored one is
    what lands.

    Returns a dict with: ``frame`` ("image" | "patient-recentred" | None
    when unverified), ``lr_switched``, ``n_landmarks``,
    ``mean_mm``/``max_mm``/``rms_mm``, ``per_tooth`` ({label: err_mm,
    delta, gt_world, landing_mm}) and ``gt_widths`` (maxilla/mandible in
    mm -- distances, so valid in either OEM frame even when the frame
    verification fails).
    """
    out = {"frame": None, "lr_switched": False, "n_landmarks": 0,
           "mean_mm": None, "max_mm": None, "rms_mm": None,
           "per_tooth": {}, "gt_widths": {}}

    om0 = _oem_name_to_label(arch_maps)
    src_w = gt_patient_pts or gt_image_pts
    if src_w:
        gl0 = {om0[n]: xyz for n, xyz in src_w.items() if n in om0}
        for arch_name, r_name, l_name in (
                ("maxilla", "Maxillary Right", "Maxillary Left"),
                ("mandible", "Mandibular Right", "Mandibular Left")):
            w = _bilateral_width(gl0, om0.get(r_name), om0.get(l_name))
            if w is not None:
                out["gt_widths"][arch_name] = w

    if seg_data is None or affine is None:
        return out

    best = None
    for src, raw in (("image", gt_image_pts), ("patient", gt_patient_pts)):
        if not raw:
            continue
        for maps, tag in ((arch_maps, "configured"),
                          (_swap_lr_labels(arch_maps), "mirrored")):
            om = _oem_name_to_label(maps)
            gl = {om[n]: xyz for n, xyz in raw.items() if n in om}
            if len(gl) < 3:
                continue
            wmap = _gt_zero_param_map(gl, affine, seg_data.shape, src)
            if len(wmap) < 3:
                continue
            h, hd = _gt_landing_distances(seg_data, affine, wmap)
            if h == len(wmap):
                key = (len(wmap), -max(d for d in hd.values()
                                       if d is not None))
                if best is None or key > best[0]:
                    best = (key, src, maps, tag, wmap, hd)
    if best is None:
        return out

    _, src, _maps, tag, wmap, hd = best
    sp = float(np.linalg.norm(affine[:3, :3], axis=0).mean())
    per = {}
    for lbl, g in wmap.items():
        p = pred_by_label.get(lbl)
        if p is None:
            continue
        p = np.asarray(p, float)
        g = np.asarray(g, float)
        if np.isnan(p).any() or np.isnan(g).any():
            continue
        d = p - g
        per[lbl] = {"err_mm": float(np.linalg.norm(d)),
                    "delta": [float(x) for x in d],
                    "gt_world": [float(x) for x in g],
                    "landing_mm": (None if hd.get(lbl) is None
                                   else float(hd[lbl] * sp))}
    eus = [v["err_mm"] for v in per.values()]
    out.update(
        frame=("image" if src == "image" else "patient-recentred"),
        lr_switched=(tag == "mirrored"),
        n_landmarks=len(per),
        per_tooth=per,
        mean_mm=float(np.mean(eus)) if eus else None,
        max_mm=float(np.max(eus)) if eus else None,
        rms_mm=float(np.sqrt(np.mean(np.square(eus)))) if eus else None,
    )
    return out


def _run_batch(cases, out_dir, arch_maps, fold, device, results_dir,
               save_seg, arch_norms, crossbite_cutoff, excess_cutoff,
               log=print, progress=None, mirror_dir=None,
               reuse_existing_masks=True, force_rerun=False):
    """Run the full pipeline on every discovered case; write the CSVs
    after EACH case so an interrupted run resumes where it stopped.

    ``batch_results.csv``    -- one row per case: widths, index,
                                diagnosis, per-tooth quality flags, GT
                                columns and the validated landmark-error
                                summary when the case shipped landmark
                                exports.
    ``batch_landmarks.csv``  -- one row per case x molar: predicted CR,
                                zero-parameter-mapped GT CR, error,
                                landing distance, quality flags.
    ``batch_cohort_stats.csv`` -- CCC / ICC(2,1) / Bland-Altman across the
                                GT-carrying cases once n >= 3.
    With ``save_seg=True``, each full labelled mask is saved next to its
    original scan (inside DICOM folders / .inv packages). Results record
    source_path, mask_path, mask_size_bytes and mask_status. Read-only scan
    folders fall back to MyDrive/CBCT_masks. Completed cases with recorded,
    present masks are skipped (both legacy CSV formats are accepted).
    A mask_ready checkpoint is written before measurement and can resume
    measurement without repeating inference. Stage timings are saved in CSV.
    With reuse_existing_masks=True, case-named masks beside the scan or in
    the Drive fallback can also be reused without a results CSV, after
    checking their grid and label data. force_rerun disables all reuse/skip.
    """
    import pandas as pd
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_csv = out_dir / "batch_results.csv"
    lm_csv = out_dir / "batch_landmarks.csv"
    _mirror_warn = {"v": False}

    done = {}
    if res_csv.exists():
        try:
            previous = pd.read_csv(res_csv, dtype={"case": str})
            done = {r["case"]: r for r in previous.to_dict("records")}
        except Exception:  # noqa: BLE001
            done = {}

    tooth_by_label = _label_to_tooth(arch_maps)
    rows_new, lm_new = [], []
    n = len(cases)

    def _mirror_csvs():
        if mirror_dir is None:
            return
        for target in (Path(mirror_dir), CSV_FALLBACK_ROOT):
            try:
                for file in (res_csv, lm_csv, out_dir / "batch_cohort_stats.csv"):
                    if file.exists():
                        _atomic_copy_file(file, target / file.name)
                if target != Path(mirror_dir) and not _mirror_warn["v"]:
                    log(f"Requested CSV mirror unavailable; mirrored instead to {target}")
                    _mirror_warn["v"] = True
                return
            except OSError as error:
                last_error = error
        log(f"WARNING: Drive CSV mirror failed ({last_error}); results remain in {out_dir}")

    for i, case in enumerate(cases):
        cid = case["case_id"]
        previous = done.get(cid, {})
        settings = f"{results_dir}|fold={fold}"
        prior_settings = previous.get("segmentation_settings")
        compatible_settings = (not isinstance(prior_settings, str)
                               or not prior_settings or prior_settings == settings)
        prior_source = previous.get("source_path")
        compatible_source = (not isinstance(prior_source, str) or not prior_source
                             or prior_source == str(Path(case["path"]).resolve()))
        if (not force_rerun and previous.get("status") == "ok"
                and compatible_settings and compatible_source
                and (not save_seg or _batch_mask_is_saved(case, previous))):
            log(f"[{i + 1}/{n}] {cid}: already in {res_csv.name} -- skipped")
            if progress:
                progress((i + 1) / n, cid, "skipped")
            continue
        log(f"[{i + 1}/{n}] {cid}: {case['kind']} -- {case['path']}")
        if cid in done and save_seg:
            log(f"[{i + 1}/{n}] {cid}: saved mask missing or unverified -- reprocessing")
        row = {"case": cid, "source_path": str(Path(case["path"]).resolve()),
               "mask_path": str(_batch_mask_path(case)) if save_seg else "",
               "mask_status": "pending" if save_seg else "disabled",
               "mask_size_bytes": None,
               "maxilla_pred": None, "mandible_pred": None,
               "index_pred": None, "maxilla_gt": None,
               "mandible_gt": None, "index_gt": None}
        row["segmentation_settings"] = settings
        started = time.perf_counter()
        lm_rows = []
        work = None
        preserve_work = False
        try:
            work = Path(tempfile.mkdtemp(prefix=f"batch_{cid}_"))
            reuse = (reuse_existing_masks and not force_rerun
                     and _batch_mask_is_saved(case, previous)
                     and previous.get("segmentation_settings") == row["segmentation_settings"])
            if reuse:
                recorded = previous.get("mask_path")
                if not isinstance(recorded, str) or not recorded:
                    recorded = previous["segmentation_path"]
                saved_mask = Path(recorded)
                seg_path = str(work / saved_mask.name)
                shutil.copyfile(saved_mask, seg_path)  # local I/O during measurement
                row.update(input_seconds=0.0, segmentation_seconds=0.0,
                           segmentation_reused=True, mask_reuse_source="results_csv",
                           mask_provenance=previous.get("mask_provenance", "recorded pipeline mask"))
                log(f"[{i + 1}/{n}] {cid}: reusing saved mask; no repeat GPU inference")
            else:
                stage = time.perf_counter()
                input_path, cid = _prepare_input_from_path(case, work, log)
                row["input_seconds"] = round(time.perf_counter() - stage, 3)
                log(f"Input preparation: {row['input_seconds']:.1f} s")
                stage = time.perf_counter()
                found = None
                if (reuse_existing_masks and not force_rerun
                        and compatible_settings and compatible_source):
                    found = _find_reusable_batch_mask(case, input_path, work, log)
                row["mask_check_seconds"] = round(time.perf_counter() - stage, 3)
                if found is not None:
                    saved_mask, seg_path = found
                    reuse = True
                    row.update(segmentation_seconds=0.0, segmentation_reused=True,
                               mask_reuse_source="scan_or_fallback_folder",
                               mask_provenance="existing mask; original model/fold not verified")
                else:
                    log("No reusable mask selected; running tooth segmentation.")
                    stage = time.perf_counter()
                    seg_path = segment_scan(
                        input_path, work_dir=work, case_id=cid,
                        fold=str(fold), device=device,
                        results_dir=results_dir, log=log)
                    row["segmentation_seconds"] = round(time.perf_counter() - stage, 3)
                    row["segmentation_reused"] = False
                    row["mask_provenance"] = "segmented in this run"
                    log(f"Segmentation including export: {row['segmentation_seconds']:.1f} s")
            if reuse:
                row["reused_mask_path"] = str(saved_mask)
            # Persist the full mask immediately, even if later measurement
            # or ground-truth validation fails.
            if save_seg:
                try:
                    stage = time.perf_counter()
                    if reuse:
                        saved_path = saved_mask
                        saved_size = saved_path.stat().st_size
                    else:
                        saved_path, saved_size = _save_batch_mask(seg_path, case, log)
                    row.update(mask_path=str(saved_path), mask_status="saved",
                               mask_size_bytes=saved_size,
                               segmentation_path=str(saved_path),
                               segmentation_original_target=str(_batch_mask_path(case)),
                               segmentation_save_location=("beside_scan" if saved_path.resolve() == _batch_mask_path(case).resolve() else "MyDrive_fallback"),
                               mask_save_seconds=round(time.perf_counter() - stage, 3))
                    # Save a recoverable checkpoint BEFORE slow measurement.
                    # If interrupted, the next run can reuse this exact mask.
                    row["status"] = "mask_ready"
                    checkpoint = (pd.read_csv(res_csv, dtype={"case": str})
                                  if res_csv.exists() else pd.DataFrame())
                    pd.concat([checkpoint, pd.DataFrame([row])], ignore_index=True
                              ).drop_duplicates(subset=["case"], keep="last").to_csv(res_csv, index=False)
                    _mirror_csvs()
                except Exception as e:
                    preserve_work = True
                    row.update(mask_status="failed", local_mask_path=str(seg_path))
                    raise OSError(
                        f"Cannot persist mask/checkpoint: {e}. "
                        f"Local mask retained at {seg_path}; check Drive write access."
                    ) from e
            log("Measuring transverse basal-bone widths...")
            stage = time.perf_counter()
            result = measure_widths(seg_path, arch_maps=arch_maps, case_id=cid)
            row["measurement_seconds"] = round(time.perf_counter() - stage, 3)
            log(f"Measurement: {row['measurement_seconds']:.1f} s")
            stage = time.perf_counter()
            row["case"] = cid
            mx = result.arch_widths_mm.get("maxilla")
            md = result.arch_widths_mm.get("mandible")
            row.update(status="ok", maxilla_pred=mx, mandible_pred=md,
                       index_pred=(None if (mx is None or md is None)
                                   else mx - md))
            if mx is not None and md is not None:
                dx = classify_transverse(
                    mx, md, crossbite_cutoff_mm=crossbite_cutoff,
                    excess_cutoff_mm=excess_cutoff)
                row["diagnosis"] = dx["label"]
                row["subclass"] = " / ".join(
                    classify_arch_width(w, a, norms=arch_norms)["label"]
                    for a, w in (("maxilla", mx), ("mandible", md)))
            pred_by_label = {
                lbl: np.asarray(t.furcation_mm, float)
                for lbl, t in result.per_tooth.items()
                if t.furcation_mm is not None}
            for lbl, t in sorted(result.per_tooth.items()):
                tooth = tooth_by_label.get(lbl, str(lbl))
                dep = t.furcation_depth_mm
                row[f"{tooth}_conf"] = t.confidence
                row[f"{tooth}_rule"] = t.rule_used
                row[f"{tooth}_depth"] = None if dep != dep else dep
                lm = {"case": cid, "tooth": tooth, "label": lbl,
                      "confidence": t.confidence, "rule": t.rule_used,
                      "depth_mm": row[f"{tooth}_depth"]}
                fp = t.furcation_mm
                if fp is not None and not np.any(np.isnan(fp)):
                    lm.update(pred_x=float(fp[0]), pred_y=float(fp[1]),
                              pred_z=float(fp[2]))
                lm_rows.append(lm)

            # --- ground truth, when the case shipped landmark exports ---
            gt_img = gt_pat = None
            if case.get("gt_image") is not None:
                gt_img, _ = _parse_oem_landmarks(
                    Path(case["gt_image"]).read_text(errors="replace"))
            if case.get("gt_patient") is not None:
                gt_pat, _ = _parse_oem_landmarks(
                    Path(case["gt_patient"]).read_text(errors="replace"))
            if gt_img or gt_pat:
                try:
                    seg_img = nib.load(str(seg_path))
                    seg_data = np.squeeze(np.asanyarray(seg_img.dataobj))
                    affine = seg_img.affine
                except Exception:  # noqa: BLE001
                    seg_data, affine = None, None
                val = _zero_param_validate(pred_by_label, gt_img, gt_pat,
                                           arch_maps, seg_data, affine)
                gmx = val["gt_widths"].get("maxilla")
                gmd = val["gt_widths"].get("mandible")
                row.update(maxilla_gt=gmx, mandible_gt=gmd,
                           index_gt=(None if (gmx is None or gmd is None)
                                     else gmx - gmd))
                for k, pk, gk in (("err_maxilla", "maxilla_pred",
                                   "maxilla_gt"),
                                  ("err_mandible", "mandible_pred",
                                   "mandible_gt"),
                                  ("err_index", "index_pred", "index_gt")):
                    if row.get(pk) is not None and row.get(gk) is not None:
                        row[k] = row[pk] - row[gk]
                row["landmark_frame"] = val["frame"] or "not verified"
                row["landmark_n"] = val["n_landmarks"]
                row["landmark_mean_mm"] = val["mean_mm"]
                row["landmark_max_mm"] = val["max_mm"]
                row["landmark_rms_mm"] = val["rms_mm"]
                row["lr_switched"] = val["lr_switched"]
                for lm in lm_rows:
                    v = val["per_tooth"].get(lm["label"])
                    if v is not None:
                        lm["gt_x"], lm["gt_y"], lm["gt_z"] = v["gt_world"]
                        lm["err_mm"] = v["err_mm"]
                        lm["point_to_tooth_mm"] = v["landing_mm"]
            else:
                row["landmark_frame"] = "no GT"

            row["validation_seconds"] = round(time.perf_counter() - stage, 3)
            log(f"[{i + 1}/{n}] {cid}: done.")
        except Exception as e:  # noqa: BLE001
            row.update(status="failed", error=str(e)[:300])
            log(f"[{i + 1}/{n}] {cid}: FAILED -- {e}")
        finally:
            row["total_seconds"] = round(time.perf_counter() - started, 3)
            log(f"[{i + 1}/{n}] {cid}: total {row['total_seconds']:.1f} s")
            if work is not None and not preserve_work:
                shutil.rmtree(work, ignore_errors=True)

        rows_new.append(row)
        lm_new.extend(lm_rows)
        _old = pd.DataFrame()
        if res_csv.exists():
            try:
                _old = pd.read_csv(res_csv, dtype={"case": str})
                _old["case"] = _old["case"].astype(str)   # CSV round-trip
            except Exception:  # noqa: BLE001             # turns '54' into 54
                pass
        pd.concat([_old, pd.DataFrame(rows_new)], ignore_index=True
                  ).drop_duplicates(subset=["case"], keep="last"
                  ).to_csv(res_csv, index=False)
        if lm_new:
            _old = pd.DataFrame()
            if lm_csv.exists():
                try:
                    _old = pd.read_csv(lm_csv, dtype={"case": str})
                    _old["case"] = _old["case"].astype(str)
                except Exception:  # noqa: BLE001
                    pass
            pd.concat([_old, pd.DataFrame(lm_new)], ignore_index=True
                      ).drop_duplicates(subset=["case", "tooth"], keep="last"
                      ).to_csv(lm_csv, index=False)
        _mirror_csvs()  # mirror AFTER current landmarks are committed
        if progress:
            progress((i + 1) / n, cid, row.get("status", "?"))

    res_df = (pd.read_csv(res_csv, dtype={"case": str}) if res_csv.exists()
              else pd.DataFrame())
    lm_df = (pd.read_csv(lm_csv, dtype={"case": str}) if lm_csv.exists()
             else pd.DataFrame())

    # Cohort agreement statistics once >= 3 cases carry ground truth.
    stats_rows = []
    if not res_df.empty:
        ok = res_df.reindex(columns=["maxilla_pred", "maxilla_gt",
                                    "mandible_pred", "mandible_gt",
                                    "index_pred", "index_gt"]).dropna(
            subset=["maxilla_pred", "maxilla_gt", "mandible_pred", "mandible_gt"])
        for name, pc, gc in (("Maxillary width", "maxilla_pred", "maxilla_gt"),
                             ("Mandibular width", "mandible_pred", "mandible_gt"),
                             ("Transverse index", "index_pred", "index_gt")):
            sub = ok.dropna(subset=[pc, gc])
            if len(sub) < 3:
                continue
            p = sub[pc].to_numpy(float)
            g = sub[gc].to_numpy(float)
            d = p - g
            stats_rows.append({
                "measurement": name, "n": len(sub),
                "CCC": _lin_ccc(p, g), "ICC_2_1": _icc_2_1(p, g),
                "bias_mm": float(d.mean()),
                "sd_diff_mm": float(d.std(ddof=1)),
                "LoA_low_mm": float(d.mean() - 1.96 * d.std(ddof=1)),
                "LoA_high_mm": float(d.mean() + 1.96 * d.std(ddof=1))})
        if "landmark_mean_mm" in res_df:
            lv = res_df.dropna(subset=["landmark_mean_mm"])
            if len(lv) >= 3:
                stats_rows.append({
                    "measurement": "CR landmark error (per-case mean)",
                    "n": len(lv), "CCC": None, "ICC_2_1": None,
                    "bias_mm": float(lv["landmark_mean_mm"].mean()),
                    "sd_diff_mm": float(lv["landmark_mean_mm"].std(ddof=1)),
                    "LoA_low_mm": None, "LoA_high_mm": None})
    if stats_rows:
        pd.DataFrame(stats_rows).to_csv(out_dir / "batch_cohort_stats.csv",
                                        index=False)
        _mirror_csvs()
    return res_df, lm_df


# --------------------------------------------------------------------------- #
#  Page config & header
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="CBCT Transverse Width",
    page_icon="🦷",
    layout="wide",
)

st.title("🦷 CBCT Transverse Basal-Bone Width")
st.caption(
    "Upload one CBCT scan → automatic tooth segmentation → predicted "
    "maxillary and mandibular transverse widths (mm), the Yonsei transverse "
    "index, and the skeletal transverse classification. No ground truth "
    "needed."
)


# --------------------------------------------------------------------------- #
#  Sidebar — model status & settings
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("⚙️ Settings")

    device = st.selectbox(
        "Compute device", ["cuda", "cpu"], index=0,
        help="Use 'cuda' on the Colab A100. 'cpu' works but is very slow.",
    )
    fold = st.text_input(
        "nnU-Net fold", value="5",
        help="Fold of the released ToothFairy2 model. Default '5'.",
    )
    with st.expander("Diagnostic cut-offs (mm)"):
        st.caption("**Transverse index** — maxillary width − mandibular width.")
        mtd_cutoff = st.number_input(
            "Crossbite cut-off (lower)", value=-2.26, step=0.01,
            format="%.2f",
            help="An index BELOW this value is skeletal crossbite "
                 "(maxillary transverse deficiency). Default −2.26 mm "
                 "= mean − 1 SD.",
        )
        excess_cutoff = st.number_input(
            "Excess cut-off (upper)", value=1.48, step=0.01, format="%.2f",
            help="An index ABOVE this value is a skeletal transverse excess "
                 "pattern. Default +1.48 mm = mean + 1 SD.",
        )

        st.caption("**Absolute arch widths** — normal range for each arch.")
        c1, c2 = st.columns(2)
        with c1:
            mx_lo = st.number_input("Maxilla min", value=45.64, step=0.01,
                                    format="%.2f",
                                    help="Below this = maxillary deficiency.")
            md_lo = st.number_input("Mandible min", value=46.30, step=0.01,
                                    format="%.2f",
                                    help="Below this = mandibular deficiency.")
        with c2:
            mx_hi = st.number_input("Maxilla max", value=51.08, step=0.01,
                                    format="%.2f",
                                    help="Above this = maxillary excess.")
            md_hi = st.number_input("Mandible max", value=51.20, step=0.01,
                                    format="%.2f",
                                    help="Above this = mandibular excess.")

    arch_norms = {
        "maxilla":  (float(mx_lo), float(mx_hi)),
        "mandible": (float(md_lo), float(md_hi)),
    }

    with st.expander("First-molar label mapping"):
        st.caption(
            "The segmentation model labels teeth 1–32. These are the "
            "first permanent molars used for the transverse measurement. "
            "The defaults below are the **handedness-corrected** mapping."
        )
        ur6 = st.number_input("Maxillary right (UR6)", value=14, step=1)
        ul6 = st.number_input("Maxillary left (UL6)",  value=6, step=1)
        lr6 = st.number_input("Mandibular right (LR6)", value=22, step=1)
        ll6 = st.number_input("Mandibular left (LL6)",  value=30, step=1)
        st.caption(
            "Why not 6 / 14 / 30 / 22? The DICOM→NIfTI conversion "
            "(LPS→RAS) flips the array handedness the pretrained "
            "ToothFairy2 model was trained on, so its raw molar labels "
            "come out mirrored: on the physical anatomy, label 6 sits on "
            "the maxillary LEFT first molar and 14 on the RIGHT "
            "(mandible: 22 RIGHT, 30 LEFT). The defaults assign each "
            "label to the tooth it physically is — verified with "
            "ground-truth landmarks mapped onto the segmentation with "
            "zero fitted parameters. The ground-truth validation below "
            "re-checks the assignment on every case and switches "
            "automatically if a scan ever arrives with the opposite "
            "handedness (a different converter). Widths are unaffected "
            "either way: a bilateral distance does not depend on which "
            "side is which."
        )
        st.caption(
            "Root counts are fixed by anatomy, not configurable: "
            f"{N_ROOTS_MAXILLARY_FIRST_MOLAR} for a maxillary first molar "
            "(mesiobuccal, distobuccal, palatal) and "
            f"{N_ROOTS_MANDIBULAR_FIRST_MOLAR} for a mandibular one "
            "(mesial, distal). The furcation is the point where that many "
            "roots have separated."
        )

    # Third element per arch = anatomical root count (see caption above).
    arch_maps = [
        ("maxilla",  {int(ur6): "UR6", int(ul6): "UL6"},
         N_ROOTS_MAXILLARY_FIRST_MOLAR),
        ("mandible", {int(lr6): "LR6", int(ll6): "LL6"},
         N_ROOTS_MANDIBULAR_FIRST_MOLAR),
    ]

    st.divider()
    st.subheader("Model status")
    results_dir = get_results_dir()
    if model_is_ready(results_dir):
        try:
            tr, pl, cf = detect_model_config(results_dir)
            st.success("Segmentation model ready")
            st.caption(f"trainer `{tr}` · plans `{pl}` · config `{cf}`")
        except Exception as e:  # noqa: BLE001
            st.success("Segmentation model ready")
            st.caption(str(e))
    else:
        st.warning(
            "Model not downloaded yet.\n\nRun the setup cell in the Colab "
            "launcher, or press the button below (one-time ~GB download)."
        )
        if st.button("⬇️ Download model now"):
            box = st.empty()
            lines: list[str] = []

            def _log(m):
                lines.append(str(m))
                box.code("\n".join(lines[-15:]))

            try:
                setup_model(results_dir, log=_log)
                st.success("Model installed. Reloading…")
                time.sleep(1)
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Download failed: {e}")


# --------------------------------------------------------------------------- #
#  Upload & run
# --------------------------------------------------------------------------- #

uploaded = st.file_uploader(
    "CBCT scan — NIfTI or DICOM",
    type=["nii", "gz", "zip", "dcm"],
    accept_multiple_files=True,
    help="Upload ONE of: a .nii/.nii.gz volume; a .zip of the DICOM folder "
         "(recommended for many slices); one multi-frame .dcm volume; "
         "or select all the .dcm slice files together.",
)
st.caption(
    "**DICOM tip:** one multi-frame `.dcm` or a complete slice series is supported. "
    "A lone 2-D slice is not a 3-D scan. For multiple volumes, the largest "
    "estimated voxel volume is tried first; zip only the intended series to select it explicitly."
)

col_run, col_info = st.columns([1, 3])
with col_run:
    run = st.button("▶️ Run analysis", type="primary",
                    disabled=not uploaded, use_container_width=True)
with col_info:
    if uploaded:
        total_mb = sum(len(f.getbuffer()) for f in uploaded) / 1e6
        if len(uploaded) == 1:
            st.caption(f"Loaded **{uploaded[0].name}** ({total_mb:.1f} MB)")
        else:
            st.caption(f"Loaded **{len(uploaded)} files** "
                       f"({total_mb:.1f} MB total)")

if run and uploaded:
    if not model_is_ready(results_dir):
        st.error(
            "The segmentation model is not installed yet. Download it from "
            "the sidebar (or the Colab setup cell) first."
        )
        st.stop()

    work_dir = Path(tempfile.mkdtemp(prefix="cbct_"))

    st.subheader("Progress")
    log_box = st.empty()
    log_lines: list[str] = []

    def log(msg):
        log_lines.append(str(msg))
        log_box.code("\n".join(log_lines[-30:]))

    t0 = time.time()
    try:
        with st.spinner("Preparing input, segmenting teeth, then measuring…"):
            # Accepts a NIfTI, a .zip of DICOMs, or many .dcm slices, and
            # returns a single .nii.gz to feed the pipeline.
            input_path, case_id = prepare_input_nifti(
                uploaded, work_dir, log=log)
            seg_path, result = run_pipeline(
                input_path,
                work_dir=work_dir,
                case_id=case_id,
                arch_maps=arch_maps,
                fold=str(fold),
                device=device,
                results_dir=results_dir,
                log=log,
            )
        # Best-effort visualisation.
        png_path = work_dir / "measurement.png"
        fig = render_measurement_figure(
            input_path, seg_path, result,
            arch_maps=arch_maps, out_png=png_path,
        )

        # Stash everything for the results section (survives reruns).
        st.session_state["result"] = result
        st.session_state["seg_path"] = str(seg_path)
        # Affine of the segmentation: needed to express the predicted CR
        # point as an image voxel index (mm -> ijk) in the results table.
        # .affine only touches the header, so this is cheap.
        try:
            st.session_state["seg_affine"] = nib.load(str(seg_path)).affine
        except Exception:  # noqa: BLE001
            st.session_state["seg_affine"] = None
        st.session_state["fig_path"] = fig
        st.session_state["case_id"] = case_id
        st.session_state["elapsed"] = time.time() - t0
        st.session_state["arch_maps"] = arch_maps
    except Exception as e:  # noqa: BLE001
        st.error(f"Pipeline failed: {e}")
        st.stop()


# ══ BATCH MODE — folder of scans → cohort CSVs ══
st.divider()
with st.expander("📁 Batch mode — process a whole folder of scans and export CSVs", expanded=False):
    st.caption("Point to a folder containing cases as NIfTI files, DICOM zips, DICOM folders, or Invivo .inv packages (subfolders OK). "
               "If a landmark CSV sits next to a case (unique or case-ID-matched), ground-truth validation runs "
               "automatically. Results are written to CSV after every case — a disconnect never loses finished work, "
               "and re-running resumes where it stopped.")
    bcol1, bcol2 = st.columns([2, 1])
    with bcol1:
        batch_root = st.text_input("Scans folder", value="/content/drive/MyDrive/CBCT_scans", key="batch_root")
        batch_out  = st.text_input("Output folder (blank → /content/batch_output, mirrored to Drive when writable)", value="", key="batch_out")
    with bcol2:
        batch_save_seg = st.checkbox(
            "Save full mask beside each original scan", value=True,
            key="batch_save_seg_beside_scan")
        batch_reuse_masks = st.checkbox(
            "Reuse existing full masks (even without results CSV)", value=True,
            key="batch_reuse_existing_masks")
        batch_force_rerun = st.checkbox(
            "Force rerun: repeat segmentation and measurements", value=False,
            key="batch_force_rerun")
        st.caption("Writes <case_id>_FULL_MASK.nii.gz beside NIfTI/ZIP scans, "
                   "inside DICOM folders, or at the root of each .inv package. "
                   "Use this folder batch mode for scans on Google Drive; "
                   "browser uploads do not include their original Drive path.")
        st.caption("Mount Drive first if the folder is in Google Drive (last cell of the notebook).")
    run_batch = st.button("▶️ Run batch", key="batch_run", use_container_width=True)
    if run_batch:
        if not model_is_ready() and (not batch_reuse_masks or batch_force_rerun):
            st.error("Model not ready. Download it before requesting new segmentation.")
        else:
            root_p = Path(batch_root.strip())
            if not root_p.exists():
                st.error(f"Folder not found: {root_p}")
            else:
                cases = _discover_batch_cases(root_p)
                if not cases:
                    exts = {}
                    for f in root_p.rglob("*"):
                        if f.is_file():
                            exts[f.suffix.lower()] = exts.get(f.suffix.lower(), 0) + 1
                    st.error(f"No usable scan cases found under {root_p}. File types present: "
                             + (", ".join(f"{k or '(none)'}×{v}" for k, v in sorted(exts.items())) or "none")
                             + ". Note: single-file .inv exports cannot be read, but .inv project PACKAGES (folders) are supported directly — the DICOM series inside is used automatically, with the native Config.inv volume as fallback.")
                else:
                    out_d = (Path(batch_out.strip()) if batch_out.strip()
                             else Path("/content/batch_output"))
                    out_d.mkdir(parents=True, exist_ok=True)
                    mirror_d = (root_p / "batch_output"
                                if str(root_p).startswith("/content/drive") else None)
                    n_gt = sum(1 for c in cases if c.get("gt_image") or c.get("gt_patient"))
                    st.info(f"Found **{len(cases)}** cases ({n_gt} with landmark CSV). Output → `{out_d}`")
                    bar = st.progress(0.0)
                    stat = st.empty()
                    loglines = []
                    logbox = st.empty()
                    def _blog(msg):
                        loglines.append(str(msg))
                        logbox.code("\n".join(loglines[-30:]), language=None)
                    def _bprog(frac, cid, status):
                        bar.progress(min(max(float(frac), 0.0), 1.0))
                        stat.caption(f"{cid} — {status}")
                    res_df, lm_df = _run_batch(
                        cases, out_d, arch_maps=arch_maps, fold=fold, device=device,
                        results_dir=results_dir, save_seg=batch_save_seg,
                        arch_norms=arch_norms, crossbite_cutoff=mtd_cutoff,
                        excess_cutoff=excess_cutoff, log=_blog, progress=_bprog, mirror_dir=mirror_d,
                        reuse_existing_masks=batch_reuse_masks, force_rerun=batch_force_rerun)
                    bar.progress(1.0); stat.caption(f"Done — {len(cases)} cases.")
                    st.session_state["batch_out_dir"] = str(out_d)
    if "batch_out_dir" in st.session_state:
        out_d = Path(st.session_state["batch_out_dir"])
        res_csv = out_d / "batch_results.csv"
        if res_csv.exists():
            res_df = pd.read_csv(res_csv)
            n_ok = int((res_df["status"] == "ok").sum()); n_err = int((res_df["status"] != "ok").sum())
            st.success(f"Batch results: **{n_ok} ok** / {n_err} error — `{res_csv}`")
            st.dataframe(res_df, use_container_width=True, hide_index=True)
            d1, d2, d3 = st.columns(3)
            d1.download_button("⬇️ batch_results.csv", res_csv.read_bytes(), "batch_results.csv", "text/csv")
            lm_csv = out_d / "batch_landmarks.csv"
            if lm_csv.exists():
                d2.download_button("⬇️ batch_landmarks.csv", lm_csv.read_bytes(), "batch_landmarks.csv", "text/csv")
            cs_csv = out_d / "batch_cohort_stats.csv"
            if cs_csv.exists():
                d3.download_button("⬇️ batch_cohort_stats.csv", cs_csv.read_bytes(), "batch_cohort_stats.csv", "text/csv")
                st.dataframe(pd.read_csv(cs_csv), use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
#  Results
# --------------------------------------------------------------------------- #

if "result" in st.session_state:
    result = st.session_state["result"]
    arch_maps = st.session_state.get("arch_maps", DEFAULT_ARCH_MAPS)

    st.divider()
    st.subheader("Predicted transverse widths")

    elapsed = st.session_state.get("elapsed")
    if elapsed:
        st.caption(f"Scan `{st.session_state.get('case_id','')}` · "
                   f"completed in {elapsed:.0f}s")

    _BADGE_COLOUR = {"deficiency": "red", "normal": "green",
                     "excess": "orange", "unknown": "gray"}

    cols = st.columns(len(arch_maps))
    for col, (name, _lm, *_) in zip(cols, arch_maps):
        w = result.arch_widths_mm.get(name)
        with col:
            if w is not None:
                st.metric(f"{name.capitalize()} width", f"{w:.2f} mm")
                sub = classify_arch_width(w, name, norms=arch_norms)
                colour = _BADGE_COLOUR.get(sub["category"], "gray")
                st.markdown(f":{colour}[**{sub['label']}**]")
                st.caption(
                    f"Normal range {sub['lower_mm']:.2f}–"
                    f"{sub['upper_mm']:.2f} mm"
                )
            else:
                st.metric(f"{name.capitalize()} width", "—")
                st.caption("Both first molars could not be measured "
                           "on this scan.")

    # --- Yonsei Transverse Index diagnosis --------------------------------
    st.subheader("Transverse diagnosis")
    mx = result.arch_widths_mm.get("maxilla")
    md = result.arch_widths_mm.get("mandible")
    if mx is not None and md is not None:
        dx = classify_transverse(
            mx, md,
            crossbite_cutoff_mm=mtd_cutoff,
            excess_cutoff_mm=excess_cutoff,
        )
        lo, hi = dx["crossbite_cutoff_mm"], dx["excess_cutoff_mm"]
        st.metric(
            "Yonsei Transverse Index  (maxilla − mandible)",
            f"{dx['index_mm']:.2f} mm",
        )

        if dx["category"] == "crossbite":
            st.error(f"🔴 **{dx['label']}**  —  index below the "
                     f"{lo:.2f} mm cut-off.")
        elif dx["category"] == "excess":
            st.warning(f"🟠 **{dx['label']}**  —  index above the "
                       f"{hi:.2f} mm cut-off.")
        else:
            st.success(f"🟢 **{dx['label']}**  —  index within "
                       f"{lo:.2f} to {hi:.2f} mm.")

        # Sub-class summary: which arch (if either) is out of range.
        mx_sub = classify_arch_width(mx, "maxilla", norms=arch_norms)
        md_sub = classify_arch_width(md, "mandible", norms=arch_norms)
        offenders = [s["label"] for s in (mx_sub, md_sub)
                     if s["category"] in ("deficiency", "excess")]
        if offenders:
            st.info("**Sub-class:** " + " · ".join(offenders))
        else:
            st.info("**Sub-class:** both arches within their normal "
                    "absolute width ranges.")

        with st.expander("Cut-offs used"):
            st.markdown(
                f"""
**Transverse index** (maxillary − mandibular width)

| Category | Rule |
| --- | --- |
| Skeletal crossbite | difference **< {lo:.2f} mm** |
| Normal transverse skeletal relationship | difference **{lo:.2f} to {hi:.2f} mm** |
| Skeletal transverse excess pattern | difference **> {hi:.2f} mm** |

**Absolute arch width**

| Sub-class | Rule |
| --- | --- |
| Maxillary deficiency | maxillary width **< {arch_norms['maxilla'][0]:.2f} mm** |
| Maxillary normal width | maxillary width **{arch_norms['maxilla'][0]:.2f}–{arch_norms['maxilla'][1]:.2f} mm** |
| Maxillary excess | maxillary width **> {arch_norms['maxilla'][1]:.2f} mm** |
| Mandibular deficiency | mandibular width **< {arch_norms['mandible'][0]:.2f} mm** |
| Mandibular normal width | mandibular width **{arch_norms['mandible'][0]:.2f}–{arch_norms['mandible'][1]:.2f} mm** |
| Mandibular excess | mandibular width **> {arch_norms['mandible'][1]:.2f} mm** |
"""
            )
            st.caption(
                "Widths are measured between the first-molar centres of "
                "resistance. The index cut-offs are mean ∓ 1 SD of normal "
                "occlusion (−0.39 ± 1.87 mm). Cut-offs are editable in the "
                "sidebar under **Diagnostic cut-offs**."
            )
    else:
        st.info(
            "The transverse index needs **both** arches measured, but one "
            "arch is missing on this scan, so a crossbite/normal call can't "
            "be made."
        )

    # Per-tooth detail (centre-of-resistance confidence).
    with st.expander("Per-tooth detail", expanded=True):
        rows = []
        name_by_label = {}
        for arch_name, lm, *_ in arch_maps:
            for lbl, tooth in lm.items():
                name_by_label[lbl] = (arch_name, tooth)
        for lbl, t in sorted(result.per_tooth.items()):
            arch_name, tooth = name_by_label.get(lbl, ("", str(lbl)))
            fp = t.furcation_mm
            fp_str = ("—" if fp is None or (hasattr(fp, "__len__")
                      and len(fp) and fp[0] != fp[0])  # NaN check
                      else f"({fp[0]:.1f}, {fp[1]:.1f}, {fp[2]:.1f})")
            depth = t.furcation_depth_mm
            rows.append({
                "Label": lbl,
                "Tooth": tooth,
                "Arch": arch_name,
                "Voxels": t.n_voxels,
                "Roots": (f"{t.n_roots_found}/{t.n_roots_expected}"
                          if t.n_roots_expected else "—"),
                "Rule": t.rule_used,
                "Depth (mm)": ("—" if depth != depth else f"{depth:.1f}"),
                "Confidence": t.confidence,
                "CR (mm)": fp_str,
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        st.caption(
            "Left/right naming is the **handedness-corrected** mapping: "
            "the DICOM→NIfTI conversion mirrors the pretrained model's "
            "raw molar labels, so on the physical anatomy label 6 is the "
            "maxillary LEFT first molar, 14 the RIGHT, 22 the mandibular "
            "RIGHT and 30 the LEFT. Uploading ground truth re-verifies "
            "this per case — the zero-parameter landing check below "
            "switches the assignment automatically if a scan was "
            "converted with the opposite handedness."
        )
        st.caption(
            "Confidence: **high** = furcation estimate agrees with an "
            "independent skeleton cross-check; **medium** = no root-side "
            "cross-check available (not a problem); **low** = the two "
            "estimates disagree — review that tooth."
        )
        st.caption(
            "Roots: separated roots found / expected for that tooth. "
            "Rule: `exact:N` = all N roots resolved, which is where the "
            "furcation centre is defined; `at_least:N` = a spurious extra "
            "component was tolerated; `relaxed:>=2` = this tooth never "
            "resolved into N roots, so the older rule was used as a "
            "fallback and its width may read slightly too wide — review it. "
            "Depth: distance from the crown's height of contour down to the "
            "furcation plane; a first molar's should be roughly 6–11 mm, and "
            "anything under ~3 mm means the estimate is still in the crown."
        )
        if any(r["Rule"] == "relaxed:>=2" for r in rows):
            st.warning(
                "One or more teeth did not resolve into their full root "
                "count, so the furcation fell back to the older rule. Those "
                "widths carry a small outward bias — check the Rule column "
                "before reporting this scan."
            )

    # --------------------------------------------------------------------- #
    #  Predicted CR point in all three coordinate frames
    # --------------------------------------------------------------------- #
    # The furcation-centre (CR) that ``molar_cr`` predicts is stored in
    # patient/world mm — the frame the NIfTI affine maps voxel indices into.
    # This block shows that predicted point three ways: verbatim as the
    # estimator emits it ("raw"), the same point expressed as an image voxel
    # index (i, j, k) via the inverse affine, and the same point in patient
    # mm. Raw and patient coincide by construction (the estimator's native
    # output IS patient mm) — showing them together makes the frame explicit
    # rather than implicit, so a predicted landmark can be matched against a
    # ground-truth table in whichever frame that table happens to use.
    with st.expander(
        "Predicted CR coordinates  —  raw · image (voxel) · patient (mm)",
        expanded=True,
    ):
        affine = st.session_state.get("seg_affine")
        if affine is None:                       # results stored before this
            seg_p = st.session_state.get("seg_path")  # feature existed
            try:
                affine = nib.load(seg_p).affine if seg_p else None
            except Exception:  # noqa: BLE001
                affine = None

        if affine is None:
            st.caption(
                "The segmentation affine is unavailable for this scan, so the "
                "image-coordinate (voxel) conversion can't be shown. Re-run "
                "the pipeline to populate it."
            )
        else:
            inv_affine = np.linalg.inv(affine)

            # Rebuild the label -> (arch, tooth) map locally so this block
            # does not depend on the per-tooth expander having run.
            _name_by_label = {}
            for _arch_name, _lm, *_ in arch_maps:
                for _lbl, _tooth in _lm.items():
                    _name_by_label[_lbl] = (_arch_name, _tooth)

            def _fmt_triplet(v, nd):
                return f"({v[0]:.{nd}f}, {v[1]:.{nd}f}, {v[2]:.{nd}f})"

            coord_rows = []
            for lbl, t in sorted(result.per_tooth.items()):
                arch_name, tooth = _name_by_label.get(lbl, ("", str(lbl)))
                fp = t.furcation_mm
                fp_arr = None if fp is None else np.asarray(fp, dtype=float)
                if fp_arr is None or fp_arr.size < 3 or np.any(np.isnan(fp_arr)):
                    coord_rows.append({
                        "Label": lbl,
                        "Tooth": tooth,
                        "Arch": arch_name,
                        "Predicted (raw xyz)": "—",
                        "Image (i, j, k voxel)": "—",
                        "Patient (x, y, z mm)": "—",
                    })
                    continue
                vox = nib.affines.apply_affine(inv_affine, fp_arr)
                coord_rows.append({
                    "Label": lbl,
                    "Tooth": tooth,
                    "Arch": arch_name,
                    # The estimator emits patient mm, so "raw" == patient.
                    "Predicted (raw xyz)": _fmt_triplet(fp_arr, 3),
                    "Image (i, j, k voxel)": _fmt_triplet(vox, 2),
                    "Patient (x, y, z mm)": _fmt_triplet(fp_arr, 3),
                })

            st.dataframe(coord_rows, use_container_width=True, hide_index=True)
            st.caption(
                "**Predicted (raw xyz)** is the CR exactly as `molar_cr` "
                "outputs it — its native frame is patient/world mm, so the "
                "**Patient (mm)** column holds the same numbers. That is not "
                "a duplicate: it documents the output frame instead of "
                "leaving it implicit. **Image (i, j, k voxel)** is the same "
                "point mapped through the inverse NIfTI affine to fractional "
                "array indices — index the volume or segmentation with it to "
                "confirm the point lands on the tooth. Voxel order follows "
                "the array axes (i → axis 0, j → axis 1, k → axis 2)."
            )

    # -----------------------------------------------------------------
    #  Ground-truth landmark validation
    # -----------------------------------------------------------------
    st.subheader("Ground-truth landmark validation")
    st.caption(
        "Upload **both** landmark exports for this scan: the **Image CS** "
        "file and the **Patient CS** file. The interface keeps the original "
        "XYZ values from each file visible and reports the validation "
        "results directly below them."
    )

    gt_col_img, gt_col_pat = st.columns(2)
    with gt_col_img:
        gt_image_file = st.file_uploader(
            "1) Ground truth — Image CS CSV",
            type=["csv"],
            key="gt_image_csv",
            help=(
                "Upload the *_Landmarks_ImageCS.csv file. "
                "This is the OEM's internal image grid (millimetres, not "
                "NIfTI voxels); the app calibrates it against the Patient "
                "CS file automatically."
            ),
        )
    with gt_col_pat:
        gt_patient_file = st.file_uploader(
            "2) Ground truth — Patient CS CSV",
            type=["csv"],
            key="gt_patient_csv",
            help=(
                "Upload the *_Landmarks_PatientCS.csv file. "
                "These coordinates are patient/world XYZ in mm."
            ),
        )

    # Parse both files independently.
    gt_image_pts, gt_image_cs = {}, "Image CS"
    gt_patient_pts, gt_patient_cs = {}, "Patient CS"

    if gt_image_file is not None:
        try:
            _txt = gt_image_file.getvalue().decode("utf-8", errors="replace")
            gt_image_pts, gt_image_cs = _parse_oem_landmarks(_txt)
        except Exception as _e:  # noqa: BLE001
            st.error(f"Could not read the Image CS CSV: {_e}")

    if gt_patient_file is not None:
        try:
            _txt = gt_patient_file.getvalue().decode("utf-8", errors="replace")
            gt_patient_pts, gt_patient_cs = _parse_oem_landmarks(_txt)
        except Exception as _e:  # noqa: BLE001
            st.error(f"Could not read the Patient CS CSV: {_e}")

    oem_map = _oem_name_to_label(arch_maps)
    tooth_by_label = _label_to_tooth(arch_maps)

    gt_image_by_label = {
        oem_map[name]: xyz for name, xyz in gt_image_pts.items()
        if name in oem_map
    }
    gt_patient_by_label = {
        oem_map[name]: xyz for name, xyz in gt_patient_pts.items()
        if name in oem_map
    }

    pred_by_label = {
        lbl: np.asarray(t.furcation_mm, float)
        for lbl, t in result.per_tooth.items()
        if t.furcation_mm is not None
    }

    # ---------------------------------------------------------------
    # Zero-parameter frame correspondence + left/right verification.
    #
    # The landmarks were digitised on THIS series, so the OEM image grid
    # and the scan's voxel grid are the same physical grid: an OEM Image
    # CS value (mm) maps to a fractional voxel index by a pure division
    # by the voxel spacing, and onward to world mm through the scan's
    # own affine -- no fitted rotation, translation, or scale of any
    # kind. (The Patient CS values work too: the OEM centres that frame
    # at the volume centre, an offset computed from the scan's shape and
    # spacing, not from the landmarks.) The mapping is verified by
    # requiring every mapped point to land on its own tooth in the
    # segmentation -- within a few voxels, since a furcation centre sits
    # in the notch BETWEEN the roots, where the mask itself can be
    # background (the "landing check").
    #
    # The same landing check verifies the left/right label assignment:
    # the DICOM->NIfTI conversion flips the array handedness the
    # pretrained model was trained on, so the model's raw molar labels
    # come out mirrored (the corrected mapping is the app default). Both
    # assignments are tested per case; if the opposite one is what lands
    # on the teeth, the validation below switches to it and says so.
    # ---------------------------------------------------------------
    gt_zero_by_label = None
    gt_zero_source = None
    gt_zero_dists = None
    _lr_switched = False
    _lm_stats = None
    _zp_seg = None
    if (gt_image_file is not None or gt_patient_file is not None) \
            and (gt_image_by_label or gt_patient_by_label):
        _zp_affine = st.session_state.get("seg_affine")
        _zp_seg_path = st.session_state.get("seg_path")
        if _zp_affine is None and _zp_seg_path:
            try:
                _zp_affine = nib.load(str(_zp_seg_path)).affine
            except Exception:  # noqa: BLE001
                _zp_affine = None
        if _zp_seg_path and os.path.exists(str(_zp_seg_path)):
            try:
                _zp_seg = np.squeeze(
                    np.asanyarray(nib.load(str(_zp_seg_path)).dataobj)
                ).astype(int)
            except Exception:  # noqa: BLE001
                _zp_seg = None

        if _zp_seg is not None and _zp_affine is not None:
            _zp_sources = []
            if gt_image_by_label:
                _zp_sources.append(("image", gt_image_pts))
            if gt_patient_by_label:
                _zp_sources.append(("patient", gt_patient_pts))
            _best = None
            for _src, _raw_pts in _zp_sources:
                for _maps, _tag in ((arch_maps, "configured"),
                                    (_swap_lr_labels(arch_maps),
                                     "mirrored")):
                    _om = _oem_name_to_label(_maps)
                    _gl = {_om[n]: xyz for n, xyz in _raw_pts.items()
                           if n in _om}
                    if len(_gl) < 3:
                        continue
                    _w = _gt_zero_param_map(
                        _gl, _zp_affine, _zp_seg.shape, _src)
                    if len(_w) < 3:
                        continue
                    _h, _hd = _gt_landing_distances(
                        _zp_seg, _zp_affine, _w)
                    # EVERY mapped point must land on (within tolerance
                    # of) its own tooth; equal-count candidates break
                    # toward the tighter landing.
                    if _h == len(_w):
                        _key = (len(_w), -max(
                            d for d in _hd.values() if d is not None))
                        if _best is None or _key > _best[0]:
                            _best = (_key, _src, _maps, _tag, _om, _w, _hd)
            if _best is not None:
                (_zkey, gt_zero_source, _zmaps, _ztag, _zom,
                 gt_zero_by_label, gt_zero_dists) = _best
                if _ztag == "mirrored":
                    _lr_switched = True
                    arch_maps = _zmaps
                    oem_map = _zom
                    tooth_by_label = _label_to_tooth(_zmaps)
                    gt_image_by_label = {
                        oem_map[n]: xyz for n, xyz in gt_image_pts.items()
                        if n in oem_map}
                    gt_patient_by_label = {
                        oem_map[n]: xyz for n, xyz in gt_patient_pts.items()
                        if n in oem_map}
                    st.warning(
                        "**Left/right labels mirrored for this case.** The "
                        "configured label map did not land the ground truth "
                        "on its teeth, but the mirrored assignment did — "
                        "this scan's conversion chain produced the opposite "
                        "handedness from the pipeline default. The "
                        "validation below uses the assignment that lands on "
                        "the teeth. The per-tooth results above were named "
                        "with the configured map, so their left/right names "
                        "are mirrored for this case; the widths are "
                        "unaffected (they are left/right symmetric)."
                    )

    # ---------------------------------------------------------------
    # Show the raw GT XYZ values from BOTH uploaded files.
    # ---------------------------------------------------------------
    if gt_image_file is not None or gt_patient_file is not None:
        st.markdown("### Ground-truth XYZ coordinates")

        raw_rows = []
        for name, code in _OEM_TOOTH_CODE.items():
            lbl = oem_map.get(name)
            if lbl is None:
                continue

            img = gt_image_by_label.get(lbl)
            pat = gt_patient_by_label.get(lbl)

            raw_rows.append({
                "Tooth": tooth_by_label.get(lbl, str(lbl)),
                "Label": lbl,
                "Image GT X": "—" if img is None else f"{img[0]:.3f}",
                "Image GT Y": "—" if img is None else f"{img[1]:.3f}",
                "Image GT Z": "—" if img is None else f"{img[2]:.3f}",
                "Patient GT X (mm)": "—" if pat is None else f"{pat[0]:.3f}",
                "Patient GT Y (mm)": "—" if pat is None else f"{pat[1]:.3f}",
                "Patient GT Z (mm)": "—" if pat is None else f"{pat[2]:.3f}",
            })

        if raw_rows:
            st.dataframe(raw_rows, use_container_width=True, hide_index=True)
            st.caption(
                "The XYZ columns above are the **original values read from the "
                "uploaded CSV files**; they are not rounded internally for "
                "validation."
            )

    # ---------------------------------------------------------------
    # Need an affine for predicted image-space coordinates.
    # ---------------------------------------------------------------
    affine = st.session_state.get("seg_affine")
    seg_path = st.session_state.get("seg_path")
    if affine is None and seg_path:
        try:
            affine = nib.load(seg_path).affine
        except Exception:  # noqa: BLE001
            affine = None

    pred_image_by_label = {}
    if affine is not None:
        try:
            inv_affine = np.linalg.inv(affine)
            for lbl, p in pred_by_label.items():
                pred_image_by_label[lbl] = nib.affines.apply_affine(inv_affine, p)
        except Exception:  # noqa: BLE001
            pred_image_by_label = {}

# ---------------------------------------------------------------
    # IMAGE-CS validation -- cross-calibrated through the Patient CS file
    # ---------------------------------------------------------------
    # The OEM "Image CS" export is NOT a voxel-index grid shared with the
    # uploaded NIfTI. On verified exports it equals the Patient CS values
    # plus a constant offset, in millimetres -- so subtracting it from
    # NIfTI voxel coordinates, as a naive implementation would, produces
    # meaningless numbers. What CAN be done, when BOTH files are present,
    # is to fit the rigid relationship between the two OEM exports from
    # the landmarks they share, express the Image CS points in patient
    # millimetres, and report what the OEM actually did (in particular:
    # whether any reorientation separates its two frames).
    if gt_image_file is not None:
        st.markdown("### Image CS validation")

        if not gt_image_by_label:
            st.warning(
                "No matching Image CS landmarks were found. Expected landmark "
                "rows named Maxillary Right/Left and Mandibular Right/Left CoR."
            )
        elif not gt_patient_by_label:
            st.info(
                "The Image CS export uses the OEM's internal image grid, "
                "which does not coincide with the uploaded scan's NIfTI voxel "
                "grid, so it cannot be interpreted on its own. Upload the "
                "matching **Patient CS** export as well: the two files share "
                "the same landmarks, which lets the app calibrate the Image "
                "CS values into patient millimetres automatically."
            )
        else:
            _common = sorted(set(gt_image_by_label) & set(gt_patient_by_label))
            if len(_common) < 3:
                st.warning(
                    "Fewer than three landmarks are shared between the Image "
                    "CS and Patient CS files, so the two exports cannot be "
                    "cross-calibrated."
                )
            else:
                _fit = _fit_rigid_transform(
                    np.array([gt_image_by_label[l] for l in _common], float),
                    np.array([gt_patient_by_label[l] for l in _common], float),
                )
                _pure_translation = (
                    _fit["angle_deg"] < 0.5
                    and abs(_fit["scale"] - 1.0) < 0.01
                )

                if _fit["rms_mm"] > 0.5:
                    st.warning(
                        f"The two exports do not describe the same four points "
                        f"(rigid-fit residual {_fit['rms_mm']:.2f} mm). Check "
                        "that both CSVs come from the same scan and the same "
                        "annotation session before trusting either file."
                    )
                elif _pure_translation:
                    st.success(
                        "The OEM applied **no reorientation**: its Image CS "
                        "and Patient CS differ only by a constant offset of "
                        f"({-_fit['t'][0]:+.2f}, {-_fit['t'][1]:+.2f}, "
                        f"{-_fit['t'][2]:+.2f}) mm (rotation "
                        f"{_fit['angle_deg']:.2f}\u00b0, scale "
                        f"{_fit['scale']:.4f}, fit residual "
                        f"{_fit['rms_mm']:.3f} mm). The two exports therefore "
                        "describe the same annotated points in one OEM "
                        "frame; how that frame relates to the scan is "
                        "established by the Patient CS validation below."
                    )
                else:
                    st.info(
                        f"Image CS \u2192 Patient CS: rotation "
                        f"{_fit['angle_deg']:.2f}\u00b0, scale "
                        f"{_fit['scale']:.4f}, translation "
                        f"({-_fit['t'][0]:+.2f}, {-_fit['t'][1]:+.2f}, "
                        f"{-_fit['t'][2]:+.2f}) mm, fit residual "
                        f"{_fit['rms_mm']:.3f} mm. The Image CS points have "
                        "been expressed in patient millimetres through this "
                        "fit; they coincide with the Patient CS values, so "
                        "the Patient CS validation below remains the "
                        "authoritative comparison."
                    )

                # Show the cross-calibration itself: Image CS points mapped
                # to patient mm vs the Patient CS values, per landmark.
                _xcal_rows = []
                for l in _common:
                    _mapped = (_fit["scale"]
                               * (np.asarray(gt_image_by_label[l], float)
                                  @ _fit["R"].T) + _fit["t"])
                    _ref = np.asarray(gt_patient_by_label[l], float)
                    _dd = _mapped - _ref
                    _xcal_rows.append({
                        "Tooth": tooth_by_label.get(l, str(l)),
                        "Label": l,
                        "Image CS \u2192 patient mm":
                            f"({_mapped[0]:.2f}, {_mapped[1]:.2f}, "
                            f"{_mapped[2]:.2f})",
                        "Patient CS (mm)":
                            f"({_ref[0]:.2f}, {_ref[1]:.2f}, {_ref[2]:.2f})",
                        "Agreement (mm)": f"{float(np.linalg.norm(_dd)):.3f}",
                    })
                st.dataframe(_xcal_rows, use_container_width=True,
                             hide_index=True)
                st.caption(
                    "Cross-calibration check: each Image CS landmark mapped "
                    "into patient millimetres through the fitted offset, "
                    "against the Patient CS value of the same landmark. "
                    "Sub-voxel agreement confirms the two exports describe "
                    "the same annotated points."
                )

    # ---------------------------------------------------------------
    # PATIENT-CS validation
    # ---------------------------------------------------------------
    if gt_patient_file is not None:
        st.markdown("### Patient CS validation")

        if not gt_patient_by_label:
            st.warning(
                "No matching Patient CS landmarks were found. Expected landmark "
                "rows named Maxillary Right/Left and Mandibular Right/Left CoR."
            )
        else:
            # Bilateral widths are frame-independent because they are distances.
            width_rows = []
            for arch_name, r_name, l_name in (
                ("Maxillary", "Maxillary Right", "Maxillary Left"),
                ("Mandibular", "Mandibular Right", "Mandibular Left"),
            ):
                r_lbl = oem_map.get(r_name)
                l_lbl = oem_map.get(l_name)

                wp = _bilateral_width(pred_by_label, r_lbl, l_lbl)
                wg = _bilateral_width(gt_patient_by_label, r_lbl, l_lbl)
                err = None if (wp is None or wg is None) else wp - wg

                width_rows.append({
                    "Measurement": f"{arch_name} width",
                    "Predicted (mm)": "—" if wp is None else f"{wp:.2f}",
                    "Ground truth (mm)": "—" if wg is None else f"{wg:.2f}",
                    "Error (mm)": "—" if err is None else f"{err:+.2f}",
                })

            rmx = _bilateral_width(
                pred_by_label, oem_map.get("Maxillary Right"),
                oem_map.get("Maxillary Left")
            )
            rmd = _bilateral_width(
                pred_by_label, oem_map.get("Mandibular Right"),
                oem_map.get("Mandibular Left")
            )
            gmx = _bilateral_width(
                gt_patient_by_label, oem_map.get("Maxillary Right"),
                oem_map.get("Maxillary Left")
            )
            gmd = _bilateral_width(
                gt_patient_by_label, oem_map.get("Mandibular Right"),
                oem_map.get("Mandibular Left")
            )

            pidx = None if (rmx is None or rmd is None) else rmx - rmd
            gidx = None if (gmx is None or gmd is None) else gmx - gmd
            ierr = None if (pidx is None or gidx is None) else pidx - gidx

            width_rows.append({
                "Measurement": "Transverse index",
                "Predicted (mm)": "—" if pidx is None else f"{pidx:.2f}",
                "Ground truth (mm)": "—" if gidx is None else f"{gidx:.2f}",
                "Error (mm)": "—" if ierr is None else f"{ierr:+.2f}",
            })

            st.markdown("**Bilateral width validation**")
            st.dataframe(width_rows, use_container_width=True, hide_index=True)

            # Absolute Patient-CS point validation, using the existing
            # no-6-DoF-fit frame reconciliation logic.
            seg_data = _zp_seg  # reuse the volume loaded for the
            # zero-parameter check above (None if that check did not run)
            if seg_data is None and seg_path and os.path.exists(seg_path):
                try:
                    seg_data = np.squeeze(
                        np.asanyarray(nib.load(seg_path).dataobj)
                    ).astype(int)
                except Exception:  # noqa: BLE001
                    seg_data = None

            if gt_zero_by_label is not None:
                _zl = sorted(set(gt_zero_by_label) & set(pred_by_label))
            else:
                _zl = []
            if len(_zl) >= 2:
                # Zero-parameter correspondence verified on this scan:
                # the GT shares the scan's origin BY CONSTRUCTION, so
                # the per-landmark comparison below is a validated
                # metric -- no frame recovery, no gate, no tolerance.
                rec = dict(mode="voxel-grid", labels=_zl,
                           gt_aligned={l: gt_zero_by_label[l] for l in _zl})
                mode = "voxel-grid"
            else:
                rec = _reconcile_frame(
                    pred_by_label,
                    gt_patient_by_label,
                    seg_data,
                    affine,
                )

                mode = rec["mode"]
                # Frames disagree -> try to recover the GT -> scan transform
                # from the scan's own segmentation (never from the predicted
                # CRs, so the recovery cannot absorb the error it measures).
                if mode in ("needs-transform", "shape-mismatch") \
                        and seg_data is not None and affine is not None:
                    recov = _recover_frame_via_teeth(
                        gt_patient_by_label, seg_data, affine)
                    recov_src = gt_patient_by_label
                    if recov is None and gt_image_by_label:
                        recov = _recover_frame_via_teeth(
                            gt_image_by_label, seg_data, affine)
                        recov_src = gt_image_by_label
                    if recov is not None:
                        rec["orig_mode"] = mode
                        rec["recovery"] = recov
                        rec["gt_aligned"] = {
                            l: recov["apply"](g) for l, g in recov_src.items()}
                        rec["tolerance_mm"] = _recovery_tolerance(
                            recov, recov_src)
                        _gf = []
                        if abs(recov["scale"] - 1.0) > 0.01:
                            _gf.append(
                                f"scale {recov['scale']:.4f} outside "
                                "[0.99, 1.01]")
                        if recov["angle_deg"] > 2.0:
                            _gf.append(
                                f"residual rotation {recov['angle_deg']:.2f}"
                                "\u00b0 > 2\u00b0")
                        if recov["rms_mm"] > 0.5:
                            _gf.append(
                                f"anchoring {recov['rms_mm']:.2f} mm > 0.5 mm")
                        if recov.get("outlier") is not None:
                            _gf.append("one landmark excluded as off-tooth")
                        rec["gate_fails"] = _gf
                        mode = "recovered-transform"
                    else:
                        rec["recovery_failed"] = True
            if mode == "voxel-grid":
                _zs = ("Image CS" if gt_zero_source == "image"
                       else "Patient CS (re-centred to the OEM image grid "
                            "from the volume geometry, not a fit)")
                _n_land = len(gt_zero_by_label)
                _dl = ""
                if gt_zero_dists:
                    _sp = float(np.linalg.norm(
                        affine[:3, :3], axis=0).mean())
                    _dl = (" Point-to-tooth distances: " + ", ".join(
                        f"{tooth_by_label.get(l, str(l))} "
                        f"{d * _sp:.2f} mm"
                        for l, d in sorted(gt_zero_dists.items())
                        if d is not None) + ".")
                st.success(
                    "**Zero-parameter frame correspondence verified.** The "
                    f"{_zs} ground truth maps onto the scan's voxel grid "
                    "by a pure division by the voxel spacing — no fitted "
                    "rotation, translation, or scale of any kind — and "
                    f"every mapped landmark lands on its own tooth "
                    f"({_n_land}/{_n_land} within landing tolerance). "
                    "GT and predictions therefore share one origin **by "
                    "construction**: the per-landmark table below is a "
                    "validated absolute-error metric, reported without a "
                    "recovery gate or tolerance." + _dl
                    + (
                        " **Left/right note:** for this case the mirrored "
                        "label assignment was the one that landed on the "
                        "teeth — the validation uses it (see the warning "
                        "above); the per-tooth tables above keep the "
                        "configured naming, which is mirrored for this "
                        "case."
                        if _lr_switched else
                        " The same landing check also confirms the "
                        "corrected left/right label assignment on this "
                        "scan."
                    )
                )
            elif mode == "identity":
                st.success(
                    "Patient CS matches the scan frame directly. "
                    "**Absolute XYZ landmark error is valid without alignment.**"
                )
            elif mode == "translation":
                t = rec["frame_offset"]
                spread = float(np.linalg.norm(rec["offset_spread"]))
                st.info(
                    f"Patient CS differs by a constant translation of "
                    f"({t[0]:+.2f}, {t[1]:+.2f}, {t[2]:+.2f}) mm "
                    f"(residual spread {spread:.2f} mm). The translation is "
                    "removed before the per-landmark distance is reported."
                )
            elif mode == "needs-transform":
                spread = float(np.linalg.norm(rec["offset_spread"]))
                st.warning(
                    f"The Patient CS frame differs from the scan frame by a "
                    f"rotation (residual spread {spread:.2f} mm), confirmed by "
                    "the pairwise-distance check: all landmark-to-landmark "
                    "distances agree, so the four-point SHAPE is intact and "
                    "only the pose differs. **Per-landmark absolute error is "
                    "not reported** because a rotation cannot be safely "
                    "separated from localization error with four near-coplanar "
                    "points. The bilateral widths above remain valid."
                )
            elif mode == "shape-mismatch":
                spread = float(np.linalg.norm(rec["offset_spread"]))
                st.warning(
                    "The predicted and ground-truth four-point **shapes "
                    "genuinely differ** -- this is real localization "
                    "disagreement on at least one landmark, not a "
                    "coordinate-frame problem (a frame rotation would leave "
                    "every pairwise distance unchanged, and they do not "
                    "match). The bilateral widths above remain valid; the "
                    "distance-by-distance breakdown below shows where the "
                    "disagreement sits."
                )
            elif mode == "recovered-transform":
                rv = rec["recovery"]
                _pp = ", ".join(
                    f"{tooth_by_label.get(l, str(l))} {d:.2f} mm"
                    for l, d in rv["per_point"].items())
                _msg = (
                    "**GT frame recovered automatically** -- the scan's own "
                    "segmentation was used as the anchor (the predicted CRs "
                    "were not used, so this cannot hide localization error). "
                    f"Axis convention: {rv['axis_map']}; residual rotation "
                    f"{rv['angle_deg']:.2f}\u00b0; scale {rv['scale']:.4f}; "
                    f"translation ({rv['t'][0]:+.1f}, {rv['t'][1]:+.1f}, "
                    f"{rv['t'][2]:+.1f}) mm. GT points anchor inside their "
                    f"teeth ({_pp})."
                )
                if rv.get("outlier") is not None:
                    _msg += (
                        " **"
                        f"{tooth_by_label.get(rv['outlier'], str(rv['outlier']))}"
                        " was excluded**: it does not land on its tooth under "
                        f"any consistent frame (off by "
                        f"{rv.get('outlier_dist_mm', float('nan')):.1f} mm) "
                        "-- check that annotation."
                    )
                _tol = rec.get("tolerance_mm")
                _gf = rec.get("gate_fails", [])
                if rec.get("orig_mode") == "shape-mismatch":
                    _msg += (
                        " The pairwise breakdown further down remains the "
                        "frame-free reference for where the disagreement "
                        "sits."
                    )
                if not _gf:
                    _msg += (
                        f" Frame-recovery tolerance for this case: "
                        f"~{_tol:.1f} mm. The per-landmark table below "
                        "**passed the pre-registered gate** (scale within "
                        "1%, residual rotation \u2264 2\u00b0, anchoring "
                        "\u2264 0.5 mm) and is a validated per-landmark "
                        "metric; still treat differences under the "
                        "tolerance as recovery artefact."
                    )
                    st.success(_msg)
                else:
                    _msg += (
                        f" Frame-recovery tolerance for this case: "
                        f"~{_tol:.1f} mm. **The per-landmark table below "
                        "is indicative only -- not a validated metric**: "
                        + "; ".join(_gf)
                        + ". Differences under the tolerance are "
                        "frame-recovery artefact, not measured error."
                    )
                    st.warning(_msg)
            else:
                st.warning(
                    "Fewer than two matched Patient CS landmarks are available, "
                    "so point-wise validation cannot be performed."
                )

            if rec.get("recovery_failed"):
                st.info(
                    "Neither the zero-parameter voxel-grid mapping nor "
                    "automatic frame recovery placed the GT points on "
                    "their teeth (or the match was ambiguous). Likely "
                    "causes: the annotated scan is not the series "
                    "uploaded here, or the OEM re-oriented / resampled "
                    "the volume on import so its grid no longer "
                    "coincides with the scan's. Falling back to "
                    "frame-invariant validation only."
                )

            if (mode == "shape-mismatch"
                    or rec.get("orig_mode") == "shape-mismatch") \
                    and rec.get("shape"):
                _sh = rec["shape"]
                _sh_rows = []
                _score = {}
                for _r in _sh:
                    _sh_rows.append({
                        "Pair": (tooth_by_label.get(_r["a"], str(_r["a"]))
                                 + " \u2013 "
                                 + tooth_by_label.get(_r["b"], str(_r["b"]))),
                        "Predicted (mm)": f"{_r['pred_mm']:.2f}",
                        "Ground truth (mm)": f"{_r['gt_mm']:.2f}",
                        "Difference (mm)": f"{_r['err_mm']:+.2f}",
                    })
                    _score[_r["a"]] = _score.get(_r["a"], 0.0) + abs(_r["err_mm"])
                    _score[_r["b"]] = _score.get(_r["b"], 0.0) + abs(_r["err_mm"])
                st.markdown("**Pairwise distance breakdown** "
                            "(frame-invariant \u2014 valid in any pose)")
                st.dataframe(_sh_rows, use_container_width=True,
                             hide_index=True)
                if _score:
                    _worst = max(_score, key=_score.get)
                    st.caption(
                        "The disagreement concentrates on "
                        f"**{tooth_by_label.get(_worst, str(_worst))}** "
                        "(largest summed distance error across its pairs). "
                        "Review that tooth's furcation estimate -- its "
                        "confidence and depth in the per-tooth table above -- "
                        "and the corresponding GT annotation, before trusting "
                        "this scan's per-landmark numbers. The arch widths "
                        "remain usable: they are the two rows of this table "
                        "with the smallest differences."
                    )

            if rec.get("gt_aligned") is not None:
                gt_aligned = rec["gt_aligned"]
                patient_rows = []
                eus = []
                signs = []

                for lbl in rec["labels"]:
                    p = np.asarray(pred_by_label[lbl], float)
                    g = np.asarray(gt_aligned[lbl], float)
                    d = p - g
                    eu = float(np.linalg.norm(d))

                    eus.append(eu)
                    signs.append(d)

                    patient_rows.append({
                        "Tooth": tooth_by_label.get(lbl, str(lbl)),
                        "Label": lbl,
                        "Predicted XYZ (mm)": (
                            f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f})"
                        ),
                        "GT XYZ (mm)": (
                            f"({g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f})"
                        ),
                        "ΔX, ΔY, ΔZ (mm)": (
                            f"({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})"
                        ),
                        "Euclidean error (mm)": f"{eu:.2f}",
                    })

                _t_suffix = ""
                if mode == "voxel-grid":
                    _t_suffix = (
                        " — GT mapped through the shared voxel grid "
                        "(zero fitted parameters)"
                    )
                elif mode == "translation":
                    _t_suffix = (
                        " — GT shifted by the detected constant frame offset"
                    )
                elif mode == "recovered-transform":
                    _t_suffix = (
                        " — GT mapped through the recovered frame transform"
                    )
                st.markdown(
                    "**Per-landmark Patient CS validation**" + _t_suffix
                )
                st.dataframe(
                    patient_rows, use_container_width=True, hide_index=True
                )

                if (mode != "recovered-transform"
                        or not rec.get("gate_fails")):
                    eus = np.asarray(eus, dtype=float)
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Mean Euclidean error", f"{eus.mean():.2f} mm")
                    c2.metric("Max Euclidean error", f"{eus.max():.2f} mm")
                    c3.metric("RMS Euclidean error", f"{np.sqrt((eus**2).mean()):.2f} mm")
                    _lm_stats = (float(eus.mean()), float(eus.max()),
                                 float(np.sqrt((eus ** 2).mean())), mode)
                elif rec.get("tolerance_mm") is not None:
                    st.caption(
                        "Summary metrics withheld: the frame-recovery gate "
                        "failed for this case (tolerance "
                        f"~{rec['tolerance_mm']:.1f} mm), so mean/RMS "
                        "per-landmark error would overstate the precision."
                    )

                if mode in ("identity", "voxel-grid"):
                    sb = np.asarray(signs).mean(axis=0)
                    st.caption(
                        f"Signed bias (mean Δ): "
                        f"({sb[0]:+.2f}, {sb[1]:+.2f}, {sb[2]:+.2f}) mm"
                    )

    # ---------------------------------------------------------------
    # Cohort statistics -- one row per validated case, accumulated on
    # disk for this session; merge prior downloads to span sessions.
    # ---------------------------------------------------------------
    if gt_patient_file is not None and gt_patient_by_label:
        _vals = (rmx, gmx, rmd, gmd, pidx, gidx)
        if all(v is not None for v in _vals):
            import pandas as _pd
            _cohort_path = "cohort_results.csv"
            _row = {
                "case": st.session_state.get("case_id", ""),
                "maxilla_pred": rmx, "maxilla_gt": gmx,
                "mandible_pred": rmd, "mandible_gt": gmd,
                "index_pred": pidx, "index_gt": gidx,
            }
            if _lm_stats is not None:
                _row.update({
                    "landmark_mean_mm": _lm_stats[0],
                    "landmark_max_mm": _lm_stats[1],
                    "landmark_rms_mm": _lm_stats[2],
                    "landmark_frame": _lm_stats[3],
                })
            _df = _pd.DataFrame([_row])
            if os.path.exists(_cohort_path):
                try:
                    _old = _pd.read_csv(_cohort_path)
                    _old = _old[_old["case"] != _row["case"]]
                    _df = _pd.concat([_old, _df], ignore_index=True)
                except Exception:  # noqa: BLE001
                    pass
            try:
                _df.to_csv(_cohort_path, index=False)
            except Exception:  # noqa: BLE001
                pass

            with st.expander(
                    f"\U0001f4ca Cohort statistics ({len(_df)} case(s))",
                    expanded=len(_df) >= 2):
                _prior = st.file_uploader(
                    "Merge a previously downloaded cohort CSV",
                    type=["csv"], key="cohort_merge")
                if _prior is not None:
                    try:
                        _dfm = _pd.concat([_pd.read_csv(_prior), _df],
                                          ignore_index=True)
                        _df = _dfm.drop_duplicates(subset=["case"],
                                                   keep="last")
                    except Exception:  # noqa: BLE001
                        st.warning("Could not read the uploaded cohort CSV.")
                st.dataframe(_df.round(3), use_container_width=True,
                             hide_index=True)
                if len(_df) >= 3:
                    _meas = (
                        ("Maxillary width", "maxilla_pred", "maxilla_gt"),
                        ("Mandibular width", "mandible_pred", "mandible_gt"),
                        ("Transverse index", "index_pred", "index_gt"),
                    )
                    for _name, _pc, _gc in _meas:
                        _p = _df[_pc].to_numpy(float)
                        _g = _df[_gc].to_numpy(float)
                        _d = _p - _g
                        _ccc = _lin_ccc(_p, _g)
                        _icc = _icc_2_1(_p, _g)
                        _bias = float(_d.mean())
                        _sd = float(_d.std(ddof=1))
                        st.markdown(
                            f"**{_name}** -- n={len(_df)} · "
                            f"CCC {_ccc:.3f} · "
                            + (f"ICC(2,1) {_icc:.3f} · "
                               if _icc is not None else "")
                            + f"Bland--Altman bias {_bias:+.2f} mm · "
                            f"95% LoA [{_bias - 1.96 * _sd:+.2f}, "
                            f"{_bias + 1.96 * _sd:+.2f}] mm "
                            f"(SD of differences {_sd:.2f} mm)"
                        )
                    import matplotlib.pyplot as _plt
                    _fig, _axes = _plt.subplots(2, 3, figsize=(14, 7))
                    for _j, (_name, _pc, _gc) in enumerate(_meas):
                        _p = _df[_pc].to_numpy(float)
                        _g = _df[_gc].to_numpy(float)
                        _d = _p - _g
                        _m = (_p + _g) / 2.0
                        _ax = _axes[0][_j]
                        _lo = min(_p.min(), _g.min()) - 1.0
                        _hi = max(_p.max(), _g.max()) + 1.0
                        _ax.plot([_lo, _hi], [_lo, _hi], "k--", lw=1)
                        _ax.scatter(_g, _p)
                        _ax.set_xlabel("Ground truth (mm)")
                        _ax.set_ylabel("Predicted (mm)")
                        _ax.set_title(_name)
                        _ax = _axes[1][_j]
                        _ax.axhline(_d.mean(), color="tab:blue")
                        _ax.axhline(_d.mean() + 1.96 * _d.std(ddof=1),
                                    color="tab:red", ls="--")
                        _ax.axhline(_d.mean() - 1.96 * _d.std(ddof=1),
                                    color="tab:red", ls="--")
                        _ax.scatter(_m, _d)
                        _ax.set_xlabel("Mean of pred & GT (mm)")
                        _ax.set_ylabel("Pred - GT (mm)")
                        _ax.set_title(f"{_name} -- Bland--Altman")
                    _fig.tight_layout()
                    st.pyplot(_fig)
                else:
                    st.caption(
                        "Validate at least 3 cases to compute CCC, ICC(2,1) "
                        "and Bland--Altman limits of agreement."
                    )
                st.download_button(
                    "\u2b07\ufe0f Download cohort CSV",
                    _df.to_csv(index=False).encode(),
                    "cohort_results.csv", "text/csv",
                    key="download_cohort",
                )

    if gt_image_file is None and gt_patient_file is None:
        st.info(
            "Upload the two GT files above to display the GT XYZ coordinates "
            "and run the landmark/width validation."
        )


    # Visualisation.
    fig_path = st.session_state.get("fig_path")
    if fig_path and os.path.exists(fig_path):
        st.subheader("Measurement overlay")
        st.image(fig_path, use_container_width=True)
        st.caption(
            "Axial slice at each arch's furcation level. The two first "
            "molars are highlighted; the yellow line is the measured "
            "transverse width between their centres of resistance."
        )

    # Download the COMPLETE segmentation mask.
    # Read the actual file bytes from disk (never a preview/array slice).
    # If nnU-Net produced an uncompressed .nii, gzip the entire NIfTI file
    # once so the browser always receives a valid .nii.gz.
    seg_path = st.session_state.get("seg_path")
    if seg_path and os.path.exists(seg_path):
        try:
            _seg_bytes = Path(seg_path).read_bytes()

            if str(seg_path).lower().endswith(".nii") and not str(seg_path).lower().endswith(".nii.gz"):
                _seg_bytes = gzip.compress(_seg_bytes, compresslevel=6)

            _download_name = f"{st.session_state.get('case_id','scan')}_seg.nii.gz"

            # Keep the exact bytes in session state so Streamlit reruns do not
            # reopen a temporary file that may have been cleaned up.
            st.session_state["seg_download_bytes"] = _seg_bytes
            st.session_state["seg_download_name"] = _download_name

            st.download_button(
                "⬇️ Download COMPLETE segmentation mask (.nii.gz)",
                data=st.session_state["seg_download_bytes"],
                file_name=st.session_state["seg_download_name"],
                mime="application/gzip",
                key="download_full_segmentation",
                help=(
                    "Downloads the complete 3-D NIfTI segmentation mask, "
                    "including every slice and every tooth label."
                ),
            )
            st.caption(
                f"Segmentation file: **{_download_name}** · "
                f"{len(_seg_bytes) / (1024**2):.2f} MB · "
                "**full 3-D mask** (not a rendered slice or preview)."
            )
        except Exception as _e:  # noqa: BLE001
            st.error(f"Could not prepare the full segmentation download: {_e}")

st.divider()
st.caption(
    "Research tool for orthodontic transverse analysis (Yonsei convention). "
    "Not a medical device; results should be reviewed by a clinician."
)
