# Image gallery

[← Overview](../README.md)


Explore the segmentation figures, measurement results, and application screenshots. **Click any figure to open it at full resolution.**

### 01 · Tooth segmentation

<table>
  <tr>
    <td align="center" width="50%">
      <a href="../fig1-segmentation-fdi-labels.png"><img src="../fig1-segmentation-fdi-labels.png" alt="Axial CBCT slice with color-coded tooth segmentation and dense model label IDs" width="420"></a><br>
      <sub><b>Figure 1 · Segmentation on the scan</b><br>Color-coded tooth masks overlaid on the CBCT.</sub>
    </td>
    <td align="center" width="50%">
      <a href="../fig2-segmentation-arch-isolated.png"><img src="../fig2-segmentation-arch-isolated.png" alt="Isolated arch segmentation against a black background, showing a separate color for each tooth" width="420"></a><br>
      <sub><b>Figure 2 · Isolated arch</b><br>Individual tooth labels ready for geometric analysis.</sub>
    </td>
  </tr>
</table>

The displayed numeric IDs are the model's dense segmentation labels, not FDI tooth numbers.

### 02 · Check the overlay

<table>
  <tr>
    <td align="center" width="50%">
      <a href="../fig3-sagittal-overlay-445.png"><img src="../fig3-sagittal-overlay-445.png" alt="Sagittal CBCT slice 445 with colored predicted tooth masks overlaid on the grayscale scan" width="420"></a><br>
      <sub><b>Figure 3 · Sagittal slice 445</b><br>Visual inspection of the segmentation.</sub>
    </td>
    <td align="center" width="50%">
      <a href="../fig4-sagittal-overlay-309.png"><img src="../fig4-sagittal-overlay-309.png" alt="Sagittal CBCT slice 309 with colored predicted tooth masks overlaid on the grayscale scan" width="420"></a><br>
      <sub><b>Figure 4 · Sagittal slice 309</b><br>A second view of mask alignment.</sub>
    </td>
  </tr>
</table>

### 03 · Inspect the furcation estimate

<p align="center">
  <a href="../fig5-furcation-diagnostics.png"><img src="../fig5-furcation-diagnostics.png" alt="Furcation diagnostics showing a crown-to-root cross-section sweep, convex-hull webbing centroid, skeleton cross-check, and per-tooth metadata" width="1000"></a>
</p>

**Figure 5 · From root separation to a 3D landmark.** The highlighted cross-section contains three separated roots. The webbing centroid supplies the furcation estimate, which is back-projected into patient coordinates and checked against the tooth skeleton.

| Diagnostic | Shown in this example |
| :--- | :--- |
| Root-count rule | `exact:3` — three expected roots found |
| Furcation depth | `7.96 mm` |
| Confidence | `medium` — no root-side skeleton branch available for comparison |
| Measurement | [Figure 6](../fig6-transverse-width-cr.png) shows the bilateral maxillary width: **39.56 mm** |

<p align="center">
  <a href="../fig6-transverse-width-cr.png"><img src="../fig6-transverse-width-cr.png" alt="Bilateral maxillary first-molar landmarks and their measured transverse width of 39.56 mm" width="1000"></a>
</p>

**Figure 6 · Bilateral transverse width.** This figure retains its original left/right label annotations. For the documented corrected mapping, see [Reference Tables](algorithm.md#reference-tables); exchanging bilateral side names does not change their distance.

### 04 · Inspect the molar in 3D

<table>
  <tr>
    <td align="center" width="50%">
      <a href="../fig7-3d-molar-furcation.png"><img src="../fig7-3d-molar-furcation.png" alt="3D UR6 tooth mesh for case170 with the red furcation point, dashed long axis, and a coordinate tooltip" width="480"></a><br>
      <sub><b>Figure 7 · Furcation coordinates</b><br>3D molar view with the landmark tooltip.</sub>
    </td>
    <td align="center" width="50%">
      <a href="../fig9-3d-molar-alternate-view.png"><img src="../fig9-3d-molar-alternate-view.png" alt="Alternate 3D view of the UR6 molar for case170, showing the three-root surface mesh, red furcation point, dashed long axis, and millimetre coordinate axes" width="480"></a><br>
      <sub><b>Figure 9 · Alternate viewing angle</b><br>A second perspective on the roots and furcation point.</sub>
    </td>
  </tr>
</table>

**Figures 7 & 9 · 3D molar and furcation landmark.** Two views of the UR6 surface mesh show the pipeline's furcation point in red and the dashed long axis. Figure 7 includes a coordinate tooltip; Figure 9 shows the tooth from another angle. These are static captures of the rotatable 3D view; click either image to open it at full resolution.

## Streamlit Interface

<p align="center">
  <a href="../fig8-streamlit-width-interface.png"><img src="../fig8-streamlit-width-interface.png" alt="Streamlit interface showing CUDA device and fold settings, editable diagnostic cut-offs, and two axial CBCT views with highlighted first molars joined by yellow width-measurement lines" width="1100"></a>
</p>

**Figure 8 · Width measurements in the application.** Two axial views display the first-molar landmarks and the yellow lines connecting them. The sidebar exposes the compute device, nnU-Net fold, diagnostic cut-offs, and absolute arch-width reference ranges.

<details>
<summary><b>Explore the settings, upload panel, and results dashboard</b></summary>

The Streamlit interface exposes the full pipeline through a browser, without writing code.

**Sidebar — Settings**
- Compute device (`cuda` / `cpu`) and nnU-Net fold selector
- **Diagnostic cut-offs** expander — every threshold from the [Reference Tables](algorithm.md#reference-tables) above is a live, editable `st.number_input`, not a hard-coded constant
- **First-molar label mapping** expander — the four label IDs are editable, with the handedness-correction rationale shown inline as UI copy
- Model status (auto-detects whether the ToothFairy2 weights are already installed, with a one-click download otherwise)

**Main panel**
- File uploader accepting `.nii`, `.nii.gz`, DICOM `.zip`, or a multi-file DICOM selection
- Live progress log while staging → segmenting → restoring the grid → measuring
- **Predicted transverse widths** and **Transverse diagnosis** — the Yonsei Index, category, and per-arch sub-classification
- **Per-tooth detail** table — voxel count, roots found/expected, rule used, furcation depth, confidence, and the raw CR coordinate for every measured tooth, plus an automatic warning banner if any tooth fell back to `relaxed:>=2`
- **Predicted CR coordinates** table shown in three frames side by side (native patient mm, and the same point reprojected to image/voxel indices) so a predicted landmark can be checked against a ground-truth table in whichever frame it uses
- **Ground-truth landmark validation** — dual CSV uploaders (Image CS / Patient CS) with the zero-parameter validation described above
- **Batch mode** expander — point it at a folder and process every case in place, with live per-case progress and CSV downloads

</details>

---

