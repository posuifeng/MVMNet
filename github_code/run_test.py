from pathlib import Path
import argparse, os, subprocess, sys
import json
import math
import hashlib

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = (
    ROOT / "data_and_weights"
    if (ROOT / "data_and_weights" / "manifest.jsonl").is_file()
    else ROOT.parent / "data_and_weights"
)


def validate_sample(row):
    """Reject incomplete labels instead of silently using loader defaults."""
    expected = {(s, v) for s in ("empty", "full") for v in ("aux1", "aux2")}
    targets = row["registration_targets"]
    if len(targets) != 4 or {(t["time_state"], t["source_key"]) for t in targets} != expected:
        raise ValueError("Expected four empty/full auxiliary-to-target labels")
    for target in targets:
        params = target["param_values"]
        if target["target_key"] != "target" or not target["head_id"]:
            raise ValueError("Invalid registration target")
        if len(params) != 6 or not all(math.isfinite(float(v)) for v in params):
            raise ValueError("Expected six finite registration parameters")
    volume = row["meta"]["volume"]
    for key in (
        "total_volume_mm3", "total_volume_m3", "grid_step_mm",
        "empty_domain_grid_samples", "usable_grid_samples",
    ):
        if not math.isfinite(float(volume[key])):
            raise ValueError(f"Non-finite volume field: {key}")
    for state in ("empty", "full"):
        for key in ("z_scale", "z_offset"):
            if not math.isfinite(float(volume[f"{state}_z_transform"][key])):
                raise ValueError("Non-finite Z transform")
        if float(volume[f"{state}_z_transform"]["z_scale"]) <= 0:
            raise ValueError("Z scale must be positive")
    for key in ("x_shift", "y_shift", "z_shift"):
        if not math.isfinite(float(volume["full_alignment_to_empty"][key])):
            raise ValueError("Non-finite alignment")
    if not math.isclose(volume["total_volume_mm3"] / 1e9, volume["total_volume_m3"], rel_tol=1e-6):
        raise ValueError("Volume units do not agree")
    if min(volume["total_volume_m3"], volume["grid_step_mm"], volume["usable_grid_samples"]) <= 0:
        raise ValueError("Volume and grid domain must be positive")
    domain = float(volume["empty_domain_grid_samples"])
    usable = float(volume["usable_grid_samples"])
    if domain <= 0 or usable > domain or not domain.is_integer() or not usable.is_integer():
        raise ValueError("Grid sample counts must be positive integers with usable <= domain")


def check_pointcloud_content(data, rows):
    """Compare XYZ multisets, including duplicate points, independently of row order."""
    import numpy as np

    seen = {}
    for row in rows:
        for state in ("empty", "full"):
            for view in ("target", "aux1", "aux2"):
                path = data / row[state][view]["points_path"]
                xyz = np.loadtxt(path, dtype=np.float64, ndmin=2)
                if xyz.shape[0] == 0 or xyz.shape[1] < 3 or not np.isfinite(xyz).all():
                    raise ValueError(f"Empty, non-finite or invalid point cloud: {path}")
                xyz = xyz[:, :3].copy()
                xyz[xyz == 0] = 0  # Canonicalize signed zero.
                xyz = xyz[np.lexsort((xyz[:, 2], xyz[:, 1], xyz[:, 0]))]
                fingerprint = hashlib.sha256(xyz.astype("<f8").tobytes()).hexdigest()
                key = (state, view, fingerprint)
                if key in seen:
                    raise ValueError(
                        f"Duplicate {state}/{view} point cloud in samples "
                        f"{seen[key]} and {row['group_id']}"
                    )
                seen[key] = row["group_id"]
    print("Content check: no duplicate same-view empty/full XYZ clouds (point order ignored).")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the 80-sample test dataset."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, help="Save evaluation to a separate directory")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate one sample through both evaluation stages.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA,
        help="Folder containing manifest.jsonl, samples, splits and weights; defaults to data_and_weights inside the code folder, then next to it",
    )
    parser.add_argument(
        "--check-content",
        action="store_true",
        help="Check data structure and compare all XYZ point clouds, ignoring point order; no inference",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Check all sample paths without running inference",
    )
    args = parser.parse_args()
    data = args.data_dir.expanduser().resolve()
    for item in ["manifest.jsonl", "splits/test.txt", "weights/best.pt"]:
        if not (data / item).is_file():
            parser.error(
                f"Missing {data / item}. Extract data_and_weights.rar and pass --data-dir pointing to the folder containing manifest.jsonl (not samples/ or the RAR file). Download: https://pan.baidu.com/s/1Am__H0IV2vVeRsX010PMTQ ; code: 8654"
            )
    if args.check_data or args.check_content:
        ids = (data / "splits/test.txt").read_text(encoding="utf-8-sig").split()
        rows = [
            json.loads(line)
            for line in (data / "manifest.jsonl")
            .read_text(encoding="utf-8-sig")
            .splitlines()
            if line.strip()
        ]
        by_id = {str(row["group_id"]): row for row in rows}
        expected_ids = [str(i) for i in range(1, 81)]
        if ids != expected_ids or set(by_id) != set(expected_ids) or len(rows) != 80:
            parser.error(
                "Expected the 80 manifest entries and matching test IDs in numeric order."
            )
        paths = set()
        for sample_id in ids:
            if sample_id not in by_id:
                parser.error(f"Sample {sample_id} missing from manifest")
            try:
                validate_sample(by_id[sample_id])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                parser.error(f"Invalid sample {sample_id}: {exc}")
            for state in ("empty", "full"):
                for view in ("target", "aux1", "aux2"):
                    for key in ("points_path", "seg_box_path", "seg_surface_path"):
                        path = data / by_id[sample_id][state][view][key]
                        if not path.is_file():
                            parser.error(f"Missing sample file: {path}")
                        paths.add(path.resolve())
        if args.check_content:
            try:
                check_pointcloud_content(data, rows)
            except (OSError, ValueError) as exc:
                parser.error(str(exc))
        print(
            f"OK: 80 samples, {len(paths)} unique point/label files, checkpoint present. Data root: {data}"
        )
        return
    output = args.output_dir.resolve() if args.output_dir else ROOT / "results" / ("smoke" if args.smoke else "full80")
    os.chdir(data)
    labels = data
    if args.smoke:
        labels = ROOT / "results" / "smoke_input"
        (labels / "splits").mkdir(parents=True, exist_ok=True)
        (labels / "manifest.jsonl").write_bytes((data / "manifest.jsonl").read_bytes())
        first = (data / "splits/test.txt").read_text().splitlines()[0]
        (labels / "splits/test.txt").write_text(first + "\n")
    common = [
        "--checkpoint",
        str(data / "weights/best.pt"),
        "--labels-dir",
        str(labels),
        "--split",
        "test",
        "--device",
        args.device,
        "--batch-size",
        "1",
        "--num-workers",
        "0",
    ]
    for script, target in [
        ("evaluate_multiview_fusion_registration_segmentation.py", "segmentation"),
        ("evaluate_multiview_volume_error.py", "volume"),
    ]:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / script),
                *common,
                "--output-dir",
                str(output / target),
            ],
            check=True,
        )
    print("Completed:", output)


if __name__ == "__main__":
    main()
