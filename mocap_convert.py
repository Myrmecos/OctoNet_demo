"""Convert an OptiTrack mocap CSV into the pose dict stored in ``*_skeleton.npy``."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

# Solved bone positions, in the order used by the released pose files.
# The skeleton root is selected separately. Ab and the finger bones are omitted.
JOINT_NAMES = [
    "Chest",
    "Neck",
    "Head",
    "LShoulder",
    "LUArm",
    "LFArm",
    "LHand",
    "RShoulder",
    "RUArm",
    "RFArm",
    "RHand",
    "LThigh",
    "LShin",
    "LFoot",
    "LToe",
    "RThigh",
    "RShin",
    "RFoot",
    "RToe",
]


def _timestamp(start64, seconds_text):
    """Capture start plus ``Time (Seconds)``, truncated to the released microsecond strings.

    Float seconds are floored to integer nanoseconds before they are added. Rounding
    the same float to microseconds is one microsecond later on many frames.
    """
    nanoseconds = int(np.floor(np.float64(seconds_text) * 1e9))
    stamp = str(start64 + np.timedelta64(nanoseconds, "ns"))
    if "." not in stamp:
        return stamp + ".000000"
    head, fraction = stamp.split(".", 1)
    return head + "." + (fraction + "000000")[:6]


def _parse_capture_start(text):
    """Parse Motive's ``YYYY-MM-DD HH.MM.SS.fff AM/PM`` start time."""
    text = text.strip()
    for fmt in ("%Y-%m-%d %I.%M.%S.%f %p", "%Y-%m-%d %I.%M.%S %p"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized Capture Start Time: {text!r}")


def _metadata(row):
    meta = {}
    for i in range(0, len(row) - 1, 2):
        key = row[i].strip()
        if key:
            meta[key] = row[i + 1].strip()
    return meta


def _bone_position_columns(type_row, name_row, kind_row):
    """Map each solved bone's short name to its three Position column indices."""
    columns = {}
    root_name = None
    i = 2
    while i < len(kind_row):
        kind = kind_row[i]
        width = 4 if kind == "Rotation" else 3
        if type_row[i] == "Bone" and kind == "Position":
            full_name = name_row[i]
            asset, _, short = full_name.partition(":")
            short = short or full_name
            if root_name is None:
                root_name = asset or short
            if short == root_name or short not in columns:
                columns[short] = list(range(i, i + width))
        i += width
    if root_name is None or root_name not in columns:
        raise ValueError("Skeleton root bone position columns were not found")
    ordered = [root_name] + JOINT_NAMES
    missing = [name for name in ordered if name not in columns]
    if missing:
        raise ValueError(f"Missing bone positions: {missing}")
    return [columns[name] for name in ordered]


def convert_mocap_csv(csv_path):
    """Return ``{'timestamps': (F,), 'positions': (F, 20, 3)}`` from one CSV."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="") as handle:
        reader = csv.reader(handle)
        meta_row = next(reader)
        header = {"type": None, "name": None, "kind": None}
        for row in reader:
            if not row or not row[0]:
                # Blank spacer, or the Type / Name / ID / Rotation rows,
                # whose first cell is empty and whose label is in column 1.
                if len(row) > 1 and row[1] == "Type":
                    header["type"] = row
                elif len(row) > 1 and row[1] == "Name":
                    header["name"] = row
                elif len(row) > 1 and row[1] == "":
                    header["kind"] = row
                continue
            if row[0] == "Frame":
                break
            raise ValueError(f"Unexpected header row starting with {row[0]!r}")
        else:
            raise ValueError(f"No frame rows in {csv_path}")

        if any(header[key] is None for key in header):
            raise ValueError(f"Incomplete OptiTrack header in {csv_path}")

        joint_cols = _bone_position_columns(header["type"], header["name"], header["kind"])
        start = _parse_capture_start(_metadata(meta_row)["Capture Start Time"])

        positions = []
        timestamps = []
        start64 = np.datetime64(start.isoformat(timespec="microseconds"))
        for row in reader:
            if not row or row[0] == "":
                continue
            positions.append([[float(row[col]) for col in cols] for cols in joint_cols])
            timestamps.append(_timestamp(start64, row[1]))

    return {
        "timestamps": np.array(timestamps, dtype=object),
        "positions": np.asarray(positions, dtype=np.float64),
    }


def save_pose(pose, npy_path):
    np.save(npy_path, pose)


def _pose_mismatch(converted, reference):
    """Return a short reason, or None when the two pose dicts match."""
    if converted["positions"].shape != reference["positions"].shape:
        return f"positions shape {converted['positions'].shape} != {reference['positions'].shape}"
    if converted["timestamps"].shape != reference["timestamps"].shape:
        return f"timestamps shape {converted['timestamps'].shape} != {reference['timestamps'].shape}"
    pos_err = np.max(np.abs(converted["positions"] - reference["positions"]))
    if pos_err != 0:
        return f"positions max abs error {pos_err}"
    if not np.array_equal(converted["timestamps"], reference["timestamps"]):
        mismatch = np.where(converted["timestamps"] != reference["timestamps"])[0]
        i = int(mismatch[0])
        return f"timestamp[{i}] {converted['timestamps'][i]!r} != {reference['timestamps'][i]!r}"
    return None


def check_directories(ori_dir, converted_dir):
    """Compare each CSV in ``ori_dir`` with ``{stem}_skeleton.npy`` in ``converted_dir``."""
    ori_dir = Path(ori_dir)
    converted_dir = Path(converted_dir)
    csv_files = sorted(ori_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {ori_dir}")

    failures = []
    for csv_path in csv_files:
        npy_path = converted_dir / f"{csv_path.stem}_skeleton.npy"
        if not npy_path.exists():
            failures.append((csv_path.name, f"missing {npy_path.name}"))
            continue
        converted = convert_mocap_csv(csv_path)
        reference = np.load(npy_path, allow_pickle=True).item()
        reason = _pose_mismatch(converted, reference)
        if reason:
            failures.append((csv_path.name, reason))
        else:
            frames = converted["positions"].shape[0]
            print(f"OK  {csv_path.name}  ->  {npy_path.name}  ({frames} frames)")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", nargs="?", help="OptiTrack CSV to convert")
    parser.add_argument("-o", "--output", help="Output .npy path")
    parser.add_argument("--check", nargs=2, metavar=("ORI_DIR", "CONVERTED_DIR"),
                        help="Compare every CSV in ORI_DIR with {stem}_skeleton.npy in CONVERTED_DIR")
    args = parser.parse_args()

    if args.check:
        failures = check_directories(args.check[0], args.check[1])
        if failures:
            for name, reason in failures:
                print(f"FAIL  {name}: {reason}")
            raise SystemExit(1)
        return

    if not args.csv_path:
        parser.error("provide a CSV path, or --check ORI_DIR CONVERTED_DIR")
    pose = convert_mocap_csv(args.csv_path)
    output = args.output or str(Path(args.csv_path).with_suffix("")) + "_skeleton.npy"
    save_pose(pose, output)
    print(f"Wrote {output}  positions {pose['positions'].shape}")


if __name__ == "__main__":
    main()
