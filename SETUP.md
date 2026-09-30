# Setup and usage

[← Project overview](./README.md) · [Application source](./app.py) · [Colab notebook](./CBCT_Width_Analysis.ipynb)

The repository includes the application extracted from `CBCT_Width_Fast_Reuse_Existing_MasksFINAL.ipynb`, plus a cleaned, self-contained copy named `CBCT_Width_Analysis.ipynb`. The measurement algorithm and original attribution are retained. Personal folder names were replaced with examples; notebook outputs and execution history were cleared. A missing module-level pandas import was added so the Streamlit batch-results panel can read its CSVs.

## Choose a workflow

| Workflow | Requirements | Start here |
| :--- | :--- | :--- |
| Direct batch in Colab | Notebook, scan folder, mounted Drive when needed | [Open in Colab](https://colab.research.google.com/github/Basma2753/CBCT-Transverse-Basal-Bone-Width-Analysis-3D-image/blob/main/CBCT_Width_Analysis.ipynb) |
| Local Streamlit interface | Python 3.11 or 3.12, Linux or WSL, analysis/UI dependencies | [Local installation](#local-installation) |
| New tooth segmentation | PyTorch, nnU-Net v2, ToothFairy2 weights; CUDA GPU for the intended GPU workflow | [Model setup](#new-segmentation-and-model-weights) |
| Reuse existing masks | Original scans and matching full tooth-label masks | [Mask reuse](#reuse-existing-masks) |

The source retains Colab paths under `/content` and uses `wget` and `unzip` for model downloads. The local commands below target Linux or WSL. Native Windows execution has not been verified.

## Google Colab

1. Open the included notebook and select a GPU runtime if new segmentation is needed.
2. Run the dependency cell. Restart the runtime if the numeric-stack check requests it, then continue with the Drive/app cells.
3. Mount Drive when your scans are stored there, and run **Write the app**.
4. Run **Download the segmentation model** only if a case will need fresh inference.
5. In **Batch configuration**, replace `/content/drive/MyDrive/CBCT_scans` with your scan directory. Set `BATCH_OUT` or keep the documented Colab default. Use `DEVICE = 'cpu'` for a measurement-only CPU batch.
6. Keep `REUSE_EXISTING_MASKS = True` and `FORCE_RERUN = False` for ordinary reuse/resume, then run the final batch cell.

The notebook installs the original CUDA 12.4 PyTorch build and pins NumPy, SciPy, and scikit-image. If the selected Colab runtime no longer supports that combination, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) and check dependency compatibility in a fresh runtime. The notebook is self-contained and does not need the repository cloned first.

## Local installation

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/Basma2753/CBCT-Transverse-Basal-Bone-Width-Analysis-3D-image.git
cd CBCT-Transverse-Basal-Bone-Width-Analysis-3D-image
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Python 3.12 may also be used with the preserved numeric pins. Dependencies other than the three numeric packages are not version-locked; the files do not represent a fully reproduced GPU environment.

Launch the interface from the repository directory:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. For measurement with existing masks, use **Batch mode**, enter explicit writable **Scans folder** and **Output folder** paths, and enable mask reuse. The single-scan upload workflow performs segmentation. If any batch mask fails validation or is missing, that case needs the segmentation stack and model as well.

## New segmentation and model weights

1. Install the PyTorch build appropriate for your OS, CUDA driver, and device using the [official installation selector](https://pytorch.org/get-started/locally/). Use the selected command inside the active virtual environment.
2. Install nnU-Net and check the environment:

   ```bash
   python -m pip install -r requirements-segmentation.txt
   python -m pip check
   python -c "import torch; print(torch.__version__, 'CUDA available:', torch.cuda.is_available())"
   ```

3. Set a writable model directory before starting Streamlit:

   ```bash
   export nnUNet_results="$PWD/nnUNet_results"
   mkdir -p "$nnUNet_results"
   python -m streamlit run app.py
   ```

4. Use the app's model-download control. The download helper requires `wget` and `unzip` on `PATH` and downloads the large [ToothSeg model archive from Zenodo](https://zenodo.org/records/14893540). The model files are separate from this repository. Keep the archive's trainer/plans/config structure intact.

The expected layout is:

```text
nnUNet_results/
└── Dataset121_ToothFairy2_Teeth/
    └── <trainer>__<plans>__<config>/
        ├── dataset.json
        ├── plans.json
        └── fold_5/
            └── checkpoint_final.pth
```

The app discovers the trainer/plans/config folder. The notebook's default fold is `5`; select a fold actually present in your downloaded model. See [nnU-Net installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md) for upstream requirements.

## Reuse existing masks

Provide the original scan and its full, multi-label segmentation. Recognized case-specific filenames include:

```text
case001.nii.gz
case001_FULL_MASK.nii.gz
```

`<case_id>_FULL_MASK.nii`, `<case_id>_seg.nii.gz`, and `<case_id>_seg.nii` are also checked. For DICOM folders and `.inv` package directories, masks belong inside the case/package folder. Colab also checks `/content/drive/MyDrive/CBCT_masks/<case_id>/`.

Masks must match the prepared scan's 3D shape and affine and use the expected dense Dataset121 labels. Empty, binary-only, corrupt, or incompatible masks are rejected. The default first-molar labels are `14` (UR6), `6` (UL6), `22` (LR6), and `30` (LL6). Geometry checks do not establish case identity or model provenance; use the corresponding scan and label mapping.

`FORCE_RERUN = True` bypasses completed-result skipping and mask reuse, clears the selected local batch CSVs, and requests new segmentation. Leave it `False` when continuing a batch. Do not run two batches into the same output directory.

## Inputs and outputs

Use complete 3D NIfTI scans, DICOM series folders/ZIPs, or single multi-frame 3D DICOM files. For Invivo projects, exported DICOM is the preferred input. The retained code contains a best-effort native `Config.inv` fallback for package directories; this is not support for arbitrary standalone `.inv` files.

Optional `ImageCS` / `PatientCS` landmark CSVs enable validation. Keep unique case identifiers when multiple studies share a directory.

| Output | Contents |
| :--- | :--- |
| `batch_results.csv` | Widths, index, quality flags, status, and timing/reuse information |
| `batch_landmarks.csv` | Predicted and optional reference landmarks |
| `batch_cohort_stats.csv` | Cohort agreement summaries when sufficient reference data exist |
| `<case_id>_FULL_MASK.nii.gz` | Full segmentation when saving is enabled |

The app records actual mask/output paths. Set an explicit local output directory in Streamlit; blank defaults target `/content/batch_output`. Colab's `/content` is temporary, so check that the requested Drive mirror was written successfully.

## Maintaining the source

The notebook's `%%writefile app.py` cell is the source of truth. To keep the standalone file synchronized after editing that cell:

```bash
python export_app.py
python export_app.py --check
```

The exporter uses only the Python standard library and does not execute the notebook or launch inference. Do not edit only the generated `app.py` and then overwrite it by running the notebook.

Before committing, clear notebook outputs and execution counts. Keep local scans, landmark exports, model weights, and generated results outside the source tree or under the ignored data/output directories. The `.gitignore` covers common extensions and generated paths; inspect staged files before publishing ZIP archives or differently named data files.

## Verification scope

Repository packaging checks cover Python syntax, notebook structure, sanitized paths/outputs, and exact notebook-to-app source matching. The supplied notebook contains reports of earlier synthetic tests and benchmarks. This source publication does not establish a new end-to-end GPU run or validate measurements on patient scans.

## Troubleshooting

- **NumPy/SciPy import errors:** use a fresh environment, install the preserved numeric versions together, and restart the Colab runtime after replacing imported libraries.
- **Model not found:** check `nnUNet_results`, the dataset/trainer directory, and the selected fold.
- **`wget` or `unzip` missing:** install those tools in the Linux/WSL environment or extract the model archive into the expected directory yourself.
- **Scan folder missing:** change the placeholder path and mount Drive before using Drive paths.
- **Existing mask ignored:** read the rejection message; check its filename, shape, affine, labels, and correspondence to the original scan.
- **One 2D DICOM slice:** supply the full slice series or a complete multi-frame 3D file.
