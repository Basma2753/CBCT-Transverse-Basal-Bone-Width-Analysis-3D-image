"""Local paths with explicit environment overrides; no Colab dependency."""
import os
from pathlib import Path


def work_root():
    return Path(os.environ.get('CBCT_WORK_DIR', Path.cwd() / '.cbct-work')).expanduser().resolve()


def model_root():
    return Path(os.environ.get('nnUNet_results', work_root() / 'nnUNet_results')).expanduser().resolve()


def mask_root():
    return Path(os.environ.get('CBCT_MASK_DIR', work_root() / 'masks')).expanduser().resolve()


def csv_root():
    return Path(os.environ.get('CBCT_CSV_DIR', work_root() / 'batch_output')).expanduser().resolve()
