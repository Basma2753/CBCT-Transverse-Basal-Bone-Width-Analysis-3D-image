# Changelog

## 0.2.0 — 2026-10-04

- Move the analysis into an installable `src/cbct_width` package; the Streamlit entry point delegates to its UI layer.
- Add a CLI for existing-mask measurement, batch processing, a synthetic demo, and explicit model setup.
- Make the package source canonical; replace notebook code generation with package imports.
- Add synthetic geometry, mask-reuse, resume, atomic-write, CLI, and Streamlit tests.
- Add Linux/Windows CI, wheel/sdist builds, and a container smoke test.
- Pin the resolved CPU/UI/test environment and Docker base-image digest.
- Use configurable local paths; remove the `wget`/`unzip` and process-wide `chdir` dependency from model setup.
- Replace result CSVs atomically and refuse to silently discard an unreadable resume file.
- Shorten the README; move the algorithm, gallery, and engineering history into documentation.
- Add a model card and separate upstream licensing notes. No project license was selected.

## 0.1.0 — 2026-09-30

- Publish the supplied single-file app, cleaned Colab notebook, setup instructions, and nine figures.
- Add the missing pandas import for batch-result display and remove personal example paths.

## Earlier revisions

The following history is retained from the original README. These entries describe earlier implementations, not newly verified clinical results.


<details>
<summary><b>View the development history</b></summary>

A condensed history from the earlier application revisions, most recent first. The package's current input and mask-reuse behavior is described in [Setup](SETUP.md).

| Version | Change |
|---|---|
| **final5_10** | Batch discovery understands Invivo `.inv` project packages directly — one case per package, real CT series selected by payload size, native `Config.inv` volume as guarded fallback |
| **final5_9** | Added interface-free direct batch run for restrictive networks; fixed direct-run to use the app's own dense label maps instead of an empty FDI fallback |
| **final5_8** | Added the batch-mode panel — folder-of-scans processing with three CSVs, crash-safe resume, auto-attached landmark CSVs |
| **final5_7** | Landing check relaxed from "exact voxel hit" to "within 8 voxels (~2.4 mm) of the point's own tooth" — a true furcation centre sits in the inter-root notch, which can itself be background |
| **final5_6** | **Left/right handedness correction** for the DICOM→NIfTI LPS→RAS flip; introduced zero-parameter ground-truth mapping as the primary validation path; automatic per-case handedness re-verification |
| **molar_cr revision 2** | Furcation search now requires the tooth's *exact* anatomical root count (3 maxillary / 2 mandibular) instead of stopping at the first 2-component split, with a crown guard against occlusal-cusp false positives; webbing region bounded by a true convex hull instead of an orientation-dependent morphological closing |

</details>

---

