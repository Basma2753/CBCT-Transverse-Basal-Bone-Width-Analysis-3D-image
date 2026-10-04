# Testing and reproducibility

[← Overview](../README.md)

## Run the suite

```bash
python -m pip install -c requirements-lock.txt -e ".[ui,dev]"
python -m pytest --cov=cbct_width --cov-report=term-missing
python -m build
```

The committed tests cover affine/crop coordinate equivalence, anisotropic physical-space PCA, translated twin molars with known 40 mm spacing, patient-space reflection/rotation/translation, absent molars, invalid/corrupt masks, reuse without inference, resume, forced reruns, discovery collisions, atomic CSV replacement failures, basic agreement-statistic cases, strict CLI JSON, imports without Streamlit/Torch, and Streamlit startup without weights.

The geometry functions and data classes were compared at the Python AST level with the pre-refactor source; their algorithm bodies were preserved. This does not validate the original scientific assumptions. Synthetic fixtures and dependency checks were run locally on Python 3.12/Windows. The GitHub workflow runs Linux and Windows tests, package builds, and a separate Docker synthetic smoke test.

## What these checks do not establish

The suite does not download model weights, invoke GPU segmentation, reproduce the reported patient-scan result, or establish cohort or clinical accuracy. No Docker executable was available on the local machine; container execution is delegated to the committed CI job. Consult that run's actual status rather than assuming success from the existence of the workflow.

The dependency constraints pin the observed CPU/UI/test environment. The Docker base image is pinned by immutable digest. The optional nnU-Net/PyTorch stack remains hardware-specific and is not fully locked or exercised here.

## Maintenance

Add regression tests before changing coordinate conventions, label maps, or root-count rules. Retain failed cases in evaluation denominators. Never replace missing clinical evidence with passing unit tests. Update the dependency constraints and base-image digest deliberately, then run the suite and build checks again.
