"""Case discovery, mask reuse, and batch result persistence."""
from __future__ import annotations
import os
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
import nibabel as nib
import re
from .config import mask_root, csv_root
from .storage import atomic_write_csv

from .validation import _bilateral_width
from .pipeline import _case_id_from_name
from .validation import _gt_landing_distances
from .validation import _gt_zero_param_map
from .validation import _icc_2_1
from .validation import _label_to_tooth
from .validation import _lin_ccc
from .validation import _oem_name_to_label
from .validation import _parse_oem_landmarks
from .validation import _swap_lr_labels
from .pipeline import classify_arch_width
from .pipeline import classify_transverse
from .pipeline import dicom_folder_to_nifti
from .pipeline import measure_widths
from .pipeline import segment_scan

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


MASK_FALLBACK_ROOT = mask_root()
CSV_FALLBACK_ROOT = csv_root()


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


def discover_cases(root):
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


def run_batch(cases, out_dir, arch_maps, fold, device, results_dir,
               save_seg=True, arch_norms=None, crossbite_cutoff=-2.26, excess_cutoff=1.48,
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
        except Exception as error:
            raise ValueError(f"Cannot read previous results {res_csv}; preserve/repair it before resuming.") from error

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
                    atomic_write_csv(pd.concat([checkpoint, pd.DataFrame([row])], ignore_index=True
                              ).drop_duplicates(subset=["case"], keep="last"), res_csv)
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
        atomic_write_csv(pd.concat([_old, pd.DataFrame(rows_new)], ignore_index=True
                  ).drop_duplicates(subset=["case"], keep="last"
                  ), res_csv)
        if lm_new:
            _old = pd.DataFrame()
            if lm_csv.exists():
                try:
                    _old = pd.read_csv(lm_csv, dtype={"case": str})
                    _old["case"] = _old["case"].astype(str)
                except Exception:  # noqa: BLE001
                    pass
            atomic_write_csv(pd.concat([_old, pd.DataFrame(lm_new)], ignore_index=True
                      ).drop_duplicates(subset=["case", "tooth"], keep="last"
                      ), lm_csv)
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
        atomic_write_csv(pd.DataFrame(stats_rows), out_dir / "batch_cohort_stats.csv")
        _mirror_csvs()
    return res_df, lm_df


# --------------------------------------------------------------------------- #
#  Page config & header
# --------------------------------------------------------------------------- #
