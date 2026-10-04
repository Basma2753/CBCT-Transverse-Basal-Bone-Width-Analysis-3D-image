"""Streamlit presentation layer; geometry and batch logic live in the package."""
from __future__ import annotations
import os
import tempfile
import time
from pathlib import Path
from .config import csv_root, work_root
from .storage import atomic_write_csv
import numpy as np
import nibabel as nib
import pandas as pd
import streamlit as st
from .pipeline import DEFAULT_ARCH_MAPS
from .geometry import N_ROOTS_MANDIBULAR_FIRST_MOLAR
from .geometry import N_ROOTS_MAXILLARY_FIRST_MOLAR
from .validation import _OEM_TOOTH_CODE
from .validation import _bilateral_width
from .batch import discover_cases
from .validation import _fit_rigid_transform
from .validation import _gt_landing_distances
from .validation import _gt_zero_param_map
from .validation import _icc_2_1
from .validation import _label_to_tooth
from .validation import _lin_ccc
from .validation import _oem_name_to_label
from .validation import _parse_oem_landmarks
from .validation import _reconcile_frame
from .validation import _recover_frame_via_teeth
from .validation import _recovery_tolerance
from .batch import run_batch
from .validation import _swap_lr_labels
from .pipeline import classify_arch_width
from .pipeline import classify_transverse
from .pipeline import detect_model_config
from .pipeline import get_results_dir
import gzip
from .pipeline import model_is_ready
from .pipeline import prepare_input_nifti
from .pipeline import render_measurement_figure
from .pipeline import run_pipeline
from .pipeline import setup_model

def main():
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
            help="Use 'cuda' with a compatible NVIDIA GPU. CPU inference can be slow.",
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
            batch_root = st.text_input("Scans folder", value=str(Path.cwd() / "scans"), key="batch_root")
            batch_out  = st.text_input("Output folder (blank uses the configured results directory)", value="", key="batch_out")
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
            st.caption("In Colab, mount Drive before selecting a Drive folder.")
        run_batch_clicked = st.button("▶️ Run batch", key="batch_run", use_container_width=True)
        if run_batch_clicked:
            if not model_is_ready() and (not batch_reuse_masks or batch_force_rerun):
                st.error("Model not ready. Download it before requesting new segmentation.")
            else:
                root_p = Path(batch_root.strip())
                if not root_p.exists():
                    st.error(f"Folder not found: {root_p}")
                else:
                    cases = discover_cases(root_p)
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
                                 else csv_root())
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
                        res_df, lm_df = run_batch(
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
                _cohort_path = str(work_root() / "cohort_results.csv")
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
                    atomic_write_csv(_df, _cohort_path)
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
