from pathlib import Path
import argparse, os, subprocess, sys
import json

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the 80-sample test dataset."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Evaluate one sample through both evaluation stages.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT.parent / "data_and_weights",
        help="Folder containing manifest.jsonl, samples, splits and weights; default: data_and_weights next to the code folder",
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
    if args.check_data:
        ids = (data / "splits/test.txt").read_text(encoding="utf-8-sig").split()
        rows = [
            json.loads(line)
            for line in (data / "manifest.jsonl")
            .read_text(encoding="utf-8-sig")
            .splitlines()
            if line.strip()
        ]
        by_id = {str(row["group_id"]): row for row in rows}
        if ids != [str(i) for i in range(1, 81)] or len(by_id) != 80 or len(rows) != 80:
            parser.error(
                "Expected 80 unique manifest entries and test IDs 1-80 in numeric order."
            )
        paths = set()
        for sample_id in ids:
            if sample_id not in by_id:
                parser.error(f"Sample {sample_id} missing from manifest")
            for state in ("empty", "full"):
                for view in ("target", "aux1", "aux2"):
                    for key in ("points_path", "seg_box_path", "seg_surface_path"):
                        path = data / by_id[sample_id][state][view][key]
                        if not path.is_file():
                            parser.error(f"Missing sample file: {path}")
                        paths.add(path.resolve())
        print(
            f"OK: 80 samples, {len(paths)} unique point/label files, checkpoint present. Data root: {data}"
        )
        return
    os.chdir(data)
    labels = data
    if args.smoke:
        labels = ROOT / "results" / "smoke_input"
        (labels / "splits").mkdir(parents=True, exist_ok=True)
        (labels / "manifest.jsonl").write_bytes((data / "manifest.jsonl").read_bytes())
        first = (data / "splits/test.txt").read_text().splitlines()[0]
        (labels / "splits/test.txt").write_text(first + "\n")
    output = ROOT / "results" / ("smoke" if args.smoke else "full80")
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
