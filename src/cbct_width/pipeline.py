from __future__ import annotations
from .config import model_root
from .geometry import CaseResult
from .geometry import N_ROOTS_MANDIBULAR_FIRST_MOLAR
from .geometry import N_ROOTS_MAXILLARY_FIRST_MOLAR
from .geometry import process_case
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
import hashlib
import urllib.request
import zipfile
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


import nibabel.processing as nibproc

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
    """Writable local model path, overridable through nnUNet_results."""
    return str(model_root())


def model_is_ready(results_dir: Optional[str] = None) -> bool:
    """Check for model metadata and at least one final checkpoint."""
    dataset = Path(results_dir or get_results_dir()) / DATASET_NAME
    return any((folder / 'plans.json').is_file()
               and (folder / 'dataset.json').is_file()
               and any(folder.glob('fold_*/checkpoint_final.pth'))
               for folder in dataset.glob('*__*__*') if folder.is_dir())


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
    trainer, plans, config = sorted(subdirs)[0].split("__")
    return trainer, plans, config


def setup_model(results_dir: Optional[str] = None, log: Callable[[str], None] = print) -> tuple[str, str, str]:
    """Explicitly download the upstream archive, verify it, and extract it.

    The MD5 is the publisher's transport-integrity checksum, not a security
    signature. Downloads use HTTPS and a temporary file. No shell utilities
    or process-wide working-directory changes are required.
    """
    root = Path(results_dir or get_results_dir()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ['nnUNet_results'] = str(root)
    if not model_is_ready(str(root)):
        archive = root / 'ToothSeg.zip'
        if not archive.is_file():
            partial = archive.with_suffix('.zip.part')
            log('Downloading the separate ToothSeg archive (~920 MB)...')
            try:
                with urllib.request.urlopen(ZENODO_URL, timeout=60) as response, partial.open('wb') as out:
                    shutil.copyfileobj(response, out)
                os.replace(partial, archive)
            finally:
                partial.unlink(missing_ok=True)
        with archive.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'md5').hexdigest()
        if digest != '5d8dd061cce9529943567aeba3271143':
            raise RuntimeError(f'Model archive checksum mismatch: {archive}. Remove the incomplete/archive copy and retry.')
        log('Extracting verified model archive...')
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                target = (root / member.filename).resolve()
                if not target.is_relative_to(root):
                    raise ValueError(f'Unsafe archive path: {member.filename}')
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError('Symlinks are not supported in the model archive.')
            zipped.extractall(root)
        dataset = root / DATASET_NAME
        matches = sorted(path for path in root.rglob(DATASET_NAME) if path.is_dir())
        if not dataset.is_dir() and len(matches) == 1:
            shutil.move(str(matches[0]), str(dataset))
        if not model_is_ready(str(root)):
            raise RuntimeError(f'Complete model metadata/checkpoint not found under {dataset}.')
    return detect_model_config(str(root))


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


