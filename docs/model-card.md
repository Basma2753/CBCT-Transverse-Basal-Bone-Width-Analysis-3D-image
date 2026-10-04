# Research system card

[← Overview](../README.md)

## Intended use

Research and engineering inspection of transverse widths derived from first-molar furcation landmarks in CBCT tooth segmentations. Outputs include estimated landmarks, bilateral widths, an index, and per-tooth quality flags. A furcation-based center-of-resistance estimate is a geometric proxy, not a patient-specific biomechanical solution.

## Components and contribution

The optional segmentation stage uses the upstream nnU-Net/ToothSeg Dataset121 model, fold 5 by default. This repository does not claim to have trained that model. Its focus is physical-space geometry, root-topology rules, landmark validation, mask reuse, and workflow integration. Data/model provenance and licensing are described [separately](third-party.md).

## Evaluation status

- **Previously reported pilot:** one scan, four first-molar landmarks, mean localization error 1.63 mm and maximum 2.30 mm under the README's zero-parameter coordinate mapping. The reference annotations and scan are not distributed here, so this result is not independently reproduced by CI.
- **Engineering verification:** deterministic synthetic volumes test software properties, including translation/reflection/rotation behavior and known bilateral spacing. These fixtures are not clinical data.
- **Not established:** cohort accuracy, confidence intervals, diagnostic sensitivity/specificity, inter-rater agreement, scanner generalization, and robustness to metal artifacts.
- **Implemented but not a completed cohort study:** CCC, ICC(2,1), and Bland–Altman summaries. The minimum of three paired cases is a numerical guard, not a validation design.

No new patient-cohort results were created as part of the engineering refactor.

## Known limitations and failure modes

Missing molars, fused/unresolved roots, segmentation mistakes, truncated anatomy, and incompatible label conventions can prevent or bias measurements. The relaxed root-count fallback is explicitly flagged. Confidence categories are heuristic QC signals; they are not calibrated probabilities or confidence intervals.

Mask geometry checks cannot prove patient identity. Source-path/fold checks do not hash scan or weight contents; replacing data in place requires deliberate recomputation. Inspect original scans, overlays, label assignments, and all low-confidence/fallback cases.

The native Invivo fallback is a retained best-effort parser, not a validated general importer. Prefer DICOM. Coordinate-recovery results must be distinguished from the primary zero-parameter validation path, because fitted alignment can affect interpretation of landmark error.

## Next evaluation

Before reporting cohort performance, freeze the pipeline version and thresholds and document the held-out case selection and reference protocol. Collect independent clinician measurements, including repeat/two-rater annotations where available. Plan sample size from the desired precision and case mix rather than treating “30 scans” as a universal adequacy rule.

Report per-arch and index errors, agreement estimates with uncertainty, failures/missing outputs, and performance by root-count rule. Keep threshold-development cases separate from evaluation cases. Describe scanner/voxel/FOV/artifact strata and any upstream training overlap. The existing code does not yet automate all these analyses.

## Data handling

No patient scans or clinician reference tables are shipped with the package, tests, or container. The synthetic demo is generated locally. A hosted demonstration should use synthetic or appropriately permitted public data and should not accept patient uploads by default. Hosting is not part of this change.

## Regulatory considerations

This is a research prototype, **not a certified medical device**. A future clinical product would require a defined intended use, risk analysis, traceable requirements, verification/validation evidence, controlled releases, and a quality process appropriate to its jurisdiction. [IEC 62304](https://webstore.iec.ch/en/publication/22794) addresses medical-device software life-cycle processes; [ISO 13485](https://www.iso.org/standard/59752.html) concerns medical-device quality management systems. Naming those standards or adding CI does not establish compliance. This repository makes no such claim.
