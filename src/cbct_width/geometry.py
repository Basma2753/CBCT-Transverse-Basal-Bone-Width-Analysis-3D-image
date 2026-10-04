"""Physical-space furcation geometry, adapted from Dr. Nadeen's original analysis."""
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


