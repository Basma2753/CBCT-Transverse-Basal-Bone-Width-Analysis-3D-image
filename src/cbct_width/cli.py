"""Command-line measurement, batch processing, and explicit model setup."""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
from pathlib import Path
import sys


def _json_value(value):
    if dataclasses.is_dataclass(value):
        return _json_value(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    if hasattr(value, 'tolist'):
        return _json_value(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(prog='cbct-width', description=__doc__)
    parser.add_argument('--version', action='version', version='cbct-width 0.2.0')
    sub = parser.add_subparsers(dest='command', required=True)
    measure = sub.add_parser('measure', help='Measure a full labelled NIfTI mask; no GPU/model required.')
    measure.add_argument('mask', type=Path)
    measure.add_argument('--output', type=Path, help='JSON destination; stdout if omitted.')
    measure.add_argument('--case-id')
    demo = sub.add_parser('demo', help='Generate and measure a synthetic four-molar phantom; no patient data.')
    demo.add_argument('--output', type=Path, required=True, help='Directory for the toy mask and measurements JSON.')
    batch = sub.add_parser('batch', help='Process original scans with optional existing-mask reuse.')
    batch.add_argument('scans', type=Path)
    batch.add_argument('--output', type=Path, required=True)
    batch.add_argument('--device', choices=('cpu', 'cuda', 'mps'), default='cpu')
    batch.add_argument('--fold', default='5')
    batch.add_argument('--model-dir', type=Path)
    batch.add_argument('--reuse-masks', action=argparse.BooleanOptionalAction, default=True)
    batch.add_argument('--save-masks', action=argparse.BooleanOptionalAction, default=True)
    batch.add_argument('--force-rerun', action='store_true', help='Recompute cases, bypassing result and mask reuse.')
    model = sub.add_parser('setup-model', help='Explicitly download and extract the separate model archive (~920 MB).')
    model.add_argument('--model-dir', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'demo':
            import nibabel as nib
            from .phantoms import make_phantom
            args.output.mkdir(parents=True, exist_ok=True)
            mask = args.output / 'synthetic_FULL_MASK.nii.gz'
            nib.save(make_phantom(), mask)
            return main(['measure', str(mask), '--case-id', 'synthetic', '--output', str(args.output / 'measurements.json')])
        if args.command == 'measure':
            from .pipeline import measure_widths
            if not args.mask.is_file():
                raise ValueError(f'Mask not found: {args.mask}')
            result = measure_widths(args.mask, case_id=args.case_id)
            payload = _json_value(result)
            mx, md = result.maxilla_width_mm, result.mandible_width_mm
            payload['transverse_index_mm'] = mx - md if mx is not None and md is not None else None
            text = json.dumps(payload, indent=2, allow_nan=False) + '\n'
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(text, encoding='utf-8')
            else:
                print(text, end='')
            return 0
        from .pipeline import DEFAULT_ARCH_MAPS, get_results_dir, setup_model
        model_dir = str(args.model_dir.resolve()) if args.model_dir else get_results_dir()
        if args.command == 'setup-model':
            setup_model(model_dir)
            return 0
        from .batch import discover_cases, run_batch
        if not args.scans.is_dir():
            raise ValueError(f'Scan directory not found: {args.scans}')
        # A nested output folder could be rediscovered as a case on reruns.
        scan_dir, output = args.scans.resolve(), args.output.resolve()
        if output == scan_dir or scan_dir in output.parents:
            raise ValueError('Use an output directory outside the scan directory.')
        cases = discover_cases(scan_dir)
        if not cases:
            raise ValueError('No supported scan volumes were found.')
        results, _ = run_batch(cases, output, arch_maps=DEFAULT_ARCH_MAPS,
            fold=args.fold, device=args.device, results_dir=model_dir,
            save_seg=args.save_masks, reuse_existing_masks=args.reuse_masks,
            force_rerun=args.force_rerun, log=print)
        failed = int((results['status'] != 'ok').sum()) if 'status' in results else len(cases)
        print(f'Finished: {len(results)} result rows; {failed} unsuccessful. Output: {output}')
        return 1 if failed else 0
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f'cbct-width: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
