# Setup and usage

[← Overview](README.md) · [Testing](docs/testing.md) · [Upstream terms](docs/third-party.md)

## Install locally

Use Python **3.12**. The core measurement path works without Streamlit, PyTorch, model weights, or a Colab runtime.

```bash
git clone https://github.com/Basma2753/CBCT-Transverse-Basal-Bone-Width-Analysis-3D-image.git
cd CBCT-Transverse-Basal-Bone-Width-Analysis-3D-image
python -m venv .venv
```

Activate the environment using `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in PowerShell. Then:

```bash
python -m pip install -c requirements-lock.txt -e .
python -m pip check
cbct-width --help
```

Install `".[ui]"` instead of `.` to include Streamlit and Matplotlib; use `".[ui,dev]"` for development. `requirements-lock.txt` pins the resolved CPU/UI/test dependencies and is used as a constraints file. GPU inference dependencies are separate and are not claimed to be reproduced by that lock.

## Synthetic demo

```bash
cbct-width demo --output demo-output
```

This writes a toy four-molar NIfTI mask and `measurements.json`. It runs the real geometry pipeline; no data or weights are downloaded. Both constructed bilateral widths are 40 mm. The shapes are software test fixtures, not validated anatomy.

## Measure an existing mask

```bash
cbct-width measure ./case001_FULL_MASK.nii.gz --output ./results/case001.json
```

The mask must use the dense Dataset121 first-molar labels: **14 → UR6, 6 → UL6, 22 → LR6, 30 → LL6**. The standalone measurement command needs the full labelled mask and its affine. The batch reuse path additionally compares it against the original scan.

Unmeasurable teeth remain explicit in the JSON: absent widths and non-finite values are serialized as `null`, with per-tooth quality flags. This is not a completed clinical measurement for every input.

## Batch processing and mask reuse

```bash
cbct-width batch ./scans --output ./results --device cpu
```

Supply original scans as complete 3D NIfTI volumes, DICOM ZIPs, DICOM folders, or single multi-frame 3D DICOM files. One ordinary 2D slice is insufficient. For Invivo projects, exported DICOM is preferred; the legacy best-effort native package fallback remains, with no claim of general `.inv` compatibility.

Case-specific masks such as `case001_FULL_MASK.nii.gz` or `case001_seg.nii.gz` go beside file scans or inside the DICOM/package folder. Uncompressed `.nii` equivalents are accepted. The reuse path checks shape, affine, finite integer label values, and the expected 0–32 range. Geometry does not prove case identity or original model provenance.

Resume is enabled by default. `--no-reuse-masks` prevents mask reuse for cases needing processing. `--force-rerun` bypasses completed-case skipping and mask reuse; `--no-save-masks` disables new mask exports. Keep each cohort's results in a separate output directory outside the scan tree. Do not run concurrent writers into one results directory.

Output CSVs are `batch_results.csv`, `batch_landmarks.csv`, and, when enough valid reference measurements exist, `batch_cohort_stats.csv`. Each local CSV uses temporary-file writing plus atomic replacement. This is per-file protection, not a multi-file transaction or a guarantee about Drive synchronization. An unreadable resume file raises an error instead of silently discarding prior results.

The CLI returns 0 for a completed command, 1 for a batch containing unsuccessful cases, and 2 for an input/setup error. Inspect missing measurements and quality flags even when the command completes.

## Configure local paths

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `CBCT_WORK_DIR` | `.cbct-work` under the launch directory | Working-data root |
| `nnUNet_results` | `<work root>/nnUNet_results` | Model directory |
| `CBCT_MASK_DIR` | `<work root>/masks` | Mask fallback for unwritable scan folders |
| `CBCT_CSV_DIR` | `<work root>/batch_output` | Default UI results / mirror fallback |

Set environment variables **before** launching Python/Streamlit. Use explicit `--output` paths for CLI batches and `--model-dir` when selecting models. Scan data and generated outputs are not part of the package.

## New segmentation and model weights

1. Install a PyTorch build for the actual hardware using the [official selector](https://pytorch.org/get-started/locally/).
2. Install the optional inference dependencies: `python -m pip install -e ".[segmentation]"` and run `python -m pip check`. Consult the [nnU-Net installation guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for upstream requirements.
3. Download the weights explicitly:

   ```bash
   cbct-width setup-model --model-dir ./models
   cbct-width batch ./scans --output ./results --model-dir ./models --device cuda --fold 5
   ```

The setup command downloads the separate **~920 MB** ToothSeg archive over HTTPS, checks the publisher's MD5 for transfer integrity, rejects unsafe extraction paths, and locates `Dataset121_ToothFairy2_Teeth`. MD5 is not a cryptographic authenticity signature. Trainer metadata and a final checkpoint must exist before the model is shown as ready. No `wget`, `unzip`, or notebook code generation is required.

The default fold is `5`, matching the original notebook. PyTorch/nnU-Net inference and the upstream weights were not exercised by the CPU synthetic suite. Do not interpret a passing package test as a successful GPU installation.

## Streamlit

```bash
python -m pip install -c requirements-lock.txt -e ".[ui]"
python -m streamlit run app.py
```

Use **Batch mode** for original scans plus saved masks. The single-scan upload workflow performs fresh segmentation. Set an explicit local output folder when needed; Colab is optional.

## Docker

```bash
docker build -t cbct-width .
docker run --rm cbct-width demo --output /work/demo
```

The image pins the official Python base by digest, installs constrained CPU/UI dependencies, and runs as a non-root user. It contains no patient scans or model weights. The Docker CI job exercises synthetic measurement; this is not a CUDA image.

For real local inputs on Linux, mount scans read-only and use your UID/GID to write a mounted output directory:

```bash
mkdir -p results
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/scans:/data:ro" -v "$PWD/results:/out" \
  cbct-width measure /data/case001_FULL_MASK.nii.gz --output /out/case001.json
```

## Optional Colab workflow

[Open the notebook](https://colab.research.google.com/github/Basma2753/CBCT-Transverse-Basal-Bone-Width-Analysis-3D-image/blob/main/CBCT_Width_Analysis.ipynb). It installs/imports the package, mounts Drive when needed, and exposes batch settings. It no longer embeds an independent copy of the application. Use a Python 3.12 runtime and a compatible GPU stack for fresh inference.

After a runtime reset, use the same Drive output directory to resume and configure `CBCT_MASK_DIR` for any existing Drive fallback masks. Check that Drive writes succeeded; local Colab storage does not survive deletion of the runtime.
