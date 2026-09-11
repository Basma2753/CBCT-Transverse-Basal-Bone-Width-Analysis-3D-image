# 🦷 CBCT Transverse Basal-Bone Width Analysis

> **AI-powered 3D CBCT analysis pipeline for automatic tooth segmentation, first-molar center-of-resistance estimation, and transverse skeletal measurements.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%20support-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![nnU-Net](https://img.shields.io/badge/Segmentation-nnU--Net%20v2-8A2BE2)](https://github.com/MIC-DKFZ/nnUNet)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Google Colab](https://img.shields.io/badge/Runtime-Google%20Colab-F9AB00?logo=googlecolab)](https://colab.research.google.com/)

## 🎥 Demo

**Interface demonstration:**  
_Add your demo video here._

Recommended GitHub format:

```text
docs/demo.mp4
```

Then add a short screen recording showing:

1. Uploading a CBCT scan
2. Running the analysis
3. Automatic tooth segmentation
4. Predicted maxillary and mandibular widths
5. Yonsei Transverse Index and classification
6. Per-tooth confidence / furcation details
7. Optional ground-truth validation
8. Batch processing and CSV export

---

## Overview

This project implements an end-to-end pipeline for extracting transverse skeletal measurements from 3D CBCT scans.

A user can upload a CBCT scan through a **Streamlit web interface**, after which the system:

```text
CBCT Scan
   │
   ├── NIfTI (.nii / .nii.gz)
   └── DICOM (.dcm / ZIP)
          │
          ▼
   Input preprocessing
          │
          ▼
   nnU-Net v2 / ToothFairy2
   Automatic tooth segmentation
          │
          ▼
   First-molar identification
          │
          ▼
   3D physical-space analysis
   • PCA long-axis estimation
   • Cross-sectional topology analysis
   • Anatomical root-count detection
   • Furcation / CR estimation
   • Skeleton cross-check
          │
          ▼
   Bilateral transverse widths
   • Maxillary width
   • Mandibular width
          │
          ▼
   Yonsei Transverse Index
          │
          ▼
   Classification + visualization
```

The pipeline does **not require ground-truth landmarks for inference**. Ground-truth CSV files can optionally be supplied for validation and cohort-level evaluation.

---

## Key Features

### 🔬 1. Automatic 3D Tooth Segmentation

The pipeline uses the released **ToothFairy2 tooth-segmentation model through nnU-Net v2** to generate a multi-label tooth segmentation from a CBCT volume.

The implementation supports:

- NIfTI input
- DICOM folders
- DICOM ZIP archives
- Multiple DICOM slices
- Automatic conversion from DICOM to NIfTI
- Full-volume segmentation restoration to the original scan grid

Nearest-neighbour interpolation is used when restoring label masks so that tooth labels remain discrete.

### 🧭 PCA-Based 3D Tooth Orientation

A key geometric step in the pipeline is estimating a stable coordinate frame for each segmented first molar. The implementation does **not** run PCA on raw voxel indices. Instead, every foreground voxel is first transformed into physical patient-space coordinates in millimetres using the NIfTI affine. PCA is then applied to that 3D point cloud.

![PCA-based tooth orientation](docs/images/pca-tooth-long-axis.svg)

**Figure 1 — PCA in physical space.** The segmented tooth mask is represented as a 3D point cloud. PCA computes three orthonormal eigenvectors ordered by eigenvalue. The dominant direction is used as the primary axis for the subsequent geometric analysis.

#### Why PCA is performed in physical space

CBCT volumes can have anisotropic voxel spacing and non-trivial orientation encoded in the affine matrix. Performing PCA directly on `(i, j, k)` voxel indices can therefore bias the estimated direction toward the more finely sampled axis. The project explicitly applies the affine first and performs the covariance/eigenvector calculation on the resulting millimetre coordinates. This is an important detail for reproducible 3D medical-image geometry.

Conceptually:

```text
Segmented tooth voxels
        │
        ▼
Voxel coordinates (i, j, k)
        │
        │  NIfTI affine
        ▼
Physical coordinates (x, y, z) in mm
        │
        ▼
Center point cloud at its centroid
        │
        ▼
3 × 3 covariance matrix
        │
        ▼
Eigenvalues + eigenvectors
        │
        ├── v1 → dominant geometric direction
        ├── v2 → orthogonal direction
        └── v3 → orthogonal direction
```

The implementation orders the eigenvectors by decreasing eigenvalue and applies a deterministic sign convention so that repeated runs produce stable axis directions. The crown-versus-root polarity is resolved separately from the arbitrary mathematical sign of an eigenvector. fileciteturn8file1L87-L94 fileciteturn8file2L148-L157

#### PCA is the starting point — not the complete anatomical decision

For molars, the direction of maximum variance is not always sufficient to define the anatomical crown-to-root direction: the crown is wide and the roots can splay. The implementation therefore evaluates the principal directions in the context of the tooth topology and selects the direction that best separates the wide crown end from the narrower root-apex end. fileciteturn8file0L55-L64

This makes the PCA frame a **geometric coordinate system for downstream analysis**, rather than assuming that the first principal component is automatically the final anatomical long axis in every molar.

### 🔎 PCA-Guided Furcation Search

Once the tooth coordinate frame has been estimated, the pipeline sweeps through the tooth along the selected long axis. At each level, voxels near the corresponding plane are projected onto the two orthogonal axes and analysed as a 2D cross-section. Connected components are then used to detect persistent root separation. fileciteturn8file0L10-L17

![PCA-guided furcation sweep](docs/images/pca-furcation-sweep.svg)

**Figure 2 — From the PCA frame to anatomical topology.** The long axis provides the sweep direction. Cross-sections are evaluated along that axis until the expected root configuration becomes persistent. This links continuous 3D geometry (PCA) with discrete anatomical topology (connected components).

The expected first-molar root counts are incorporated into the decision:

| Tooth | Expected roots |
|---|---:|
| Maxillary first molar | 3 |
| Mandibular first molar | 2 |

A single noisy cross-section is not sufficient; the implementation requires persistent component separation to reduce sensitivity to segmentation noise. fileciteturn8file2L172-L194

### 🧮 PCA in the Overall Measurement Pipeline

PCA is therefore used as a **geometric foundation** for the first-molar analysis:

1. Extract the binary tooth mask from the nnU-Net segmentation.
2. Convert foreground voxels from voxel indices to physical millimetre coordinates.
3. Compute the centroid and three orthogonal principal directions.
4. Resolve the anatomical crown-to-root orientation.
5. Sweep along the selected axis.
6. Detect persistent root components in cross-sections.
7. Localize the furcation region.
8. Estimate the furcation/center-of-resistance point.
9. Use the bilateral points to calculate maxillary and mandibular transverse widths.

The project explicitly documents that the physical-coordinate transformation is essential because anisotropic spacing can otherwise distort the recovered direction. fileciteturn8file0L6-L17

### 🦷 2. First-Molar Center-of-Resistance Estimation

For each relevant first molar, the system estimates a furcation-based center of resistance in physical millimetre coordinates.

The analysis is performed in **physical space rather than raw voxel space**, allowing the affine matrix and voxel spacing to account for anisotropic resolution and image orientation.

The algorithm includes:

- 3D tooth-mask cleaning
- Physical-coordinate transformation
- PCA-based tooth long-axis estimation
- Cross-sectional rasterization
- Connected-component analysis
- Persistent root separation detection
- Furcation localization
- Convex-hull based webbing analysis
- 3D skeletonization as an independent cross-check

### 🧠 3. Anatomically Aware Root Detection

The furcation detector uses the expected anatomical root count:

| Arch | First molar | Expected roots |
|---|---|---:|
| Maxilla | First permanent molar | 3 |
| Mandible | First permanent molar | 2 |

For maxillary molars, the algorithm waits for the three-root configuration instead of stopping at an earlier two-component split.

A crown guard prevents occlusal cusp separation from being incorrectly interpreted as a furcation.

If a segmentation does not resolve the expected roots, the system reports the fallback rule rather than silently treating the result as equally reliable.

### 📐 4. Transverse Width Measurement

The transverse width of each arch is calculated as the Euclidean distance between the two first-molar furcation/CR points:

```text
Maxillary width = distance(UR6_CR, UL6_CR)

Mandibular width = distance(LR6_CR, LL6_CR)
```

All measurements are reported in **millimetres**.

### 📊 5. Yonsei Transverse Index

The system calculates:

```text
Yonsei Transverse Index
    = Maxillary width − Mandibular width
```

The interface reports the index together with the configured diagnostic cut-offs and classification.

Absolute arch-width ranges are also used to provide an additional per-arch sub-classification.

> **Important:** The classification thresholds are configurable in the Streamlit sidebar and should be interpreted according to the clinical/reference protocol used for a given study.

### 🧪 6. Ground-Truth Validation

The interface supports optional OEM landmark CSV exports:

- Image Coordinate System (Image CS)
- Patient Coordinate System (Patient CS)

The validation pipeline can:

- Parse landmark exports
- Match landmarks to first-molar labels
- Check coordinate-frame correspondence
- Verify left/right label assignment
- Perform zero-parameter voxel-grid mapping when the grids correspond
- Detect frame mismatches
- Recover a frame transform when necessary
- Compare predicted and reference CR coordinates
- Calculate Euclidean landmark errors
- Report mean, maximum, and RMS error
- Compare bilateral widths against ground truth

This separates **coordinate-frame problems** from genuine localization disagreement instead of automatically fitting predictions to ground truth.

### 📈 7. Cohort-Level Evaluation

For multiple validated cases, the system can calculate:

- Maxillary width agreement
- Mandibular width agreement
- Transverse-index agreement
- Concordance Correlation Coefficient (CCC)
- ICC(2,1)
- Bland–Altman bias
- Bland–Altman limits of agreement
- Per-case landmark error summaries

### 📁 8. Batch Processing

The application includes a batch-processing interface for analyzing an entire folder of scans.

Supported case sources include:

- NIfTI volumes
- DICOM ZIP files
- DICOM folders
- Invivo `.inv` project packages

Batch processing produces:

```text
batch_results.csv
batch_landmarks.csv
batch_cohort_stats.csv
```

The batch engine is designed to be **crash-safe**:

- Results are written after each processed case
- Completed cases are skipped when the process is restarted
- Failed cases can be retried
- Landmark files are automatically associated with matching cases

---

## Streamlit Interface

The application provides a browser-based interface with:

### Input

Upload:

- `.nii`
- `.nii.gz`
- DICOM ZIP
- DICOM slices

### Processing

The interface displays live progress while:

1. Preparing the input
2. Converting DICOM when necessary
3. Running nnU-Net segmentation
4. Restoring the segmentation to the original CBCT grid
5. Detecting molar furcations
6. Computing transverse widths
7. Generating diagnostic measurements

### Results

The results dashboard includes:

- Maxillary transverse width
- Mandibular transverse width
- Per-arch classification
- Yonsei Transverse Index
- Transverse skeletal classification
- Per-tooth confidence
- Root-count information
- Furcation depth
- Predicted CR coordinates
- Measurement visualization
- Optional ground-truth comparison
- Downloadable CSV results

---

## Technical Highlights

### Physical-space geometry

Voxel indices are transformed through the NIfTI affine:

```python
patient_xyz = nib.affines.apply_affine(affine, voxel_ijk)
```

This allows the geometry calculations to operate in millimetres rather than assuming isotropic voxels.

### PCA-based tooth orientation

The tooth mask is converted to physical coordinates and PCA is used to estimate:

- Long axis
- Two orthogonal cross-sectional axes

This is important because performing PCA directly on anisotropic voxel coordinates can distort the estimated tooth orientation.

### Furcation detection

The algorithm sweeps along the tooth's long axis and evaluates cross-sectional connected components.

A persistent root split is required instead of relying on a single noisy slice.

### Independent skeleton check

A 3D morphological skeleton is used as an independent check of the detected furcation.

Large disagreement between the primary estimate and the skeleton branch point lowers the reported confidence.

### Rotation-independent webbing estimate

The furcation region uses a convex hull rather than a fixed morphological closing operation, reducing orientation-dependent behaviour in the cross-sectional geometry.

---

## Technology Stack

| Component | Technology |
|---|---|
| Programming | Python |
| Deep learning | PyTorch |
| Medical image segmentation | nnU-Net v2 / ToothFairy2 |
| 3D medical imaging | NIfTI, DICOM |
| Medical image I/O | NiBabel, SimpleITK, pydicom |
| Numerical computing | NumPy, SciPy |
| Image processing | scikit-image |
| Visualization | Matplotlib |
| Web interface | Streamlit |
| GPU environment | Google Colab / NVIDIA A100 |
| Batch analytics | Pandas |
| Statistical evaluation | CCC, ICC(2,1), Bland–Altman |

---

## Running the Project

### Option 1 — Google Colab

The provided notebook is designed as a self-contained Colab launcher.

1. Open the notebook in Google Colab.
2. Select an **A100 GPU** runtime.
3. Run the installation/setup cells.
4. Mount Google Drive if batch scans are stored there.
5. Run the application launcher.
6. Open the generated Streamlit tunnel URL.
7. Upload a CBCT scan and run the analysis.

The notebook generates a single self-contained `app.py`, so the Streamlit application does not depend on separate local pipeline modules.

### Option 2 — Local Streamlit

After preparing the required model environment:

```bash
streamlit run app.py
```

The application is then available through the local Streamlit server.

---

## Example Output

A successful analysis produces measurements similar to:

```text
Maxillary transverse width  : XX.XX mm
Mandibular transverse width : XX.XX mm

Yonsei Transverse Index     : XX.XX mm

Diagnosis:
Normal transverse skeletal relationship
```

The exact values depend on the input CBCT scan.

---

## Validation Philosophy

A major focus of this project is **measurement validity rather than simply producing a number**.

The validation workflow explicitly distinguishes:

```text
Coordinate-frame mismatch
        ↓
Can the frames be reconciled?
        ↓
Does the landmark shape agree?
        ↓
Does the mapped landmark land on its intended tooth?
        ↓
If valid → calculate localization error
```

This is intended to prevent a common failure mode in 3D medical-image evaluation: treating a coordinate-system mismatch as model localization error.

---

## Project Structure

A recommended GitHub repository structure is:

```text
CBCT-Transverse-Width/
│
├── README.md
├── app.py
├── CBCT_Width_Colab.ipynb
│
├── docs/
│   └── demo.mp4
│
├── examples/
│   └── README.md
│
├── results/
│   └── README.md
│
└── requirements.txt
```

For a public repository, avoid committing:

- Patient CBCT scans
- DICOM files
- Patient identifiers
- Ground-truth files containing identifiable information
- Large model checkpoints when they are available from the original public release

---

## What I Built

This project demonstrates an end-to-end **medical computer vision / AI engineering workflow**:

- 3D medical image preprocessing
- DICOM/NIfTI handling
- Deep-learning inference
- nnU-Net integration
- 3D geometric computer vision
- Anatomical topology analysis
- Coordinate-system reasoning
- Quantitative validation
- Statistical agreement analysis
- Streamlit application development
- Batch processing
- Robust error handling
- GPU-based deployment through Google Colab

Rather than being only a research notebook, the project turns the research pipeline into an interactive application that can be demonstrated and used through a web interface.

---

## Research & Clinical Context

The furcation-center approach follows the transverse-analysis methodology implemented in this project. The codebase also documents the anatomical assumptions, coordinate conventions, validation strategy, and limitations directly alongside the implementation.

This repository is intended as a **research and engineering project**, not as a certified medical device or a replacement for professional clinical assessment.

---

## Demo

📹 **Add your interface demo video here.**

For a job-focused portfolio, the strongest demo is a **60–90 second screen recording**:

```text
0:00 — Upload CBCT
0:10 — Run analysis
0:20 — Segmentation / processing progress
0:35 — Maxillary + mandibular measurements
0:45 — Yonsei Transverse Index
0:55 — Per-tooth confidence / CR coordinates
1:05 — Ground-truth validation
1:20 — Batch CSV export
```

---

## Author

**[Your Name]**

Computer Science | AI & Computer Vision | Medical Imaging

Interested in:

- Machine Learning Engineering
- Computer Vision
- 3D Medical Imaging
- Deep Learning
- AI Systems & Deployment

---

## ⭐ Why this project matters for an AI/ML portfolio

This project combines **research, computer vision, 3D geometry, deep learning inference, medical-image processing, validation, and deployment** in one workflow.

It demonstrates the ability to move from:

**raw medical data → ML inference → geometric reasoning → quantitative validation → usable application.**
