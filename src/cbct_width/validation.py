"""Landmark coordinate reconciliation and agreement statistics."""
from __future__ import annotations
import numpy as np
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

