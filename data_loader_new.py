"""Load one prepared OctoNet recording for visualization.

The recording is the user-1 ``airdrum`` take under ``downloaded_octonet_oneuser/``.
Sensor streams are returned as arrays. Depth stays on disk until a time slice
is requested, because the full camera stream is large. RGB video is not loaded.
"""

import ast
import csv
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from mocap_convert import convert_mocap_csv

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "downloaded_octonet_oneuser"
DEFAULT_METADATA = ROOT / "cut_manual.csv"

# PNG grayscale 0..255 maps onto this distance and temperature range.
DEPTH_MAX_METERS = 10.0
THERMAL_MIN_C = 15.0
THERMAL_MAX_C = 50.0

_NTP_START = re.compile(r"NTP Time:\s*\[(.*?)\]: Start Recording")
_COLUMN_MODALITY = re.compile(r"^node_(\d+)_(.+)_data_path$")


def load_recording(
    dataset_root=DEFAULT_DATASET,
    metadata_csv=DEFAULT_METADATA,
    user_id="1",
    activity="airdrum",
    timestamp="20240519182852",
    node_id=None,
):
    """Load every stream for one metadata row.

    ``timestamp`` matches ``new_timestamp`` in ``cut_manual.csv``. Depth cameras
    are returned as paths; call :func:`iter_segments` to decode frames for one cut.
    RGB video is not opened. ``node_id`` keeps that sensor node. A modality that
    does not have this node, and has only one node, is still included.
    """
    dataset_root = Path(dataset_root)
    keep_node = None if node_id is None else int(node_id)
    row = _metadata_row(metadata_csv, str(user_id), activity, timestamp)
    recording = {
        "user_id": row["user_id"],
        "activity": row["activity"],
        "timestamp": row.get("new_timestamp", timestamp),
        "segments": segment_bounds(row["cut_timestamps"]),
        "missing": [],
    }
    pending = {}

    for column, value in row.items():
        if not value:
            continue
        if column == "mocap_data_path":
            recording["mocap"] = _load_mocap(dataset_root / value)
            continue
        if column == "imu_data_path":
            recording["imu"] = _load_imu(dataset_root / value)
            continue
        match = _COLUMN_MODALITY.match(column)
        if not match:
            continue
        sensor_node = int(match.group(1))
        modality = match.group(2)
        pending.setdefault(modality, {})[sensor_node] = dataset_root / value

    for modality, nodes in pending.items():
        if keep_node is None:
            chosen = nodes
        elif keep_node in nodes:
            chosen = {keep_node: nodes[keep_node]}
        elif len(nodes) == 1:
            chosen = nodes
        else:
            continue
        for sensor_node, path in chosen.items():
            if not path.exists():
                recording["missing"].append(str(path))
                continue
            stream = _LOADERS[modality](path)
            recording.setdefault(modality, {})[sensor_node] = stream
    return recording


def segment_bounds(cut_timestamps):
    """Middle activity windows, dropping the first two cuts and the last one.

    This is the same rule as ``OctoNet/dataset_loader.py``: with boundary times
    ``b[0] .. b[N-1]``, the kept intervals are ``[b[2], b[3])`` through
    ``[b[N-3], b[N-2])``.
    """
    if isinstance(cut_timestamps, str):
        cut_timestamps = ast.literal_eval(cut_timestamps)
    bounds = [_parse_time(ts) for ts in cut_timestamps]
    if len(bounds) < 5:
        return []
    return [(bounds[i], bounds[i + 1]) for i in range(2, len(bounds) - 2)]


def iter_segments(recording):
    """Yield ``(start, end, clip)`` for each middle cut.

    ``clip`` has the same modality keys as ``recording``, sliced to ``[start, end)``.
    Depth is float32 meters.
    """
    for start, end in recording["segments"]:
        yield start, end, _slice_recording(recording, start, end)


def load_depth_frames(paths):
    """Read depth PNGs and map 0..255 onto 0..10 meters. Shape ``(T, H, W)``."""
    frames = [_read_gray(path) for path in paths]
    if not frames:
        return np.zeros((0, 480, 640), dtype=np.float32)
    stack = np.stack(frames).astype(np.float32)
    return stack / 255.0 * DEPTH_MAX_METERS


def load_rgb_frames(video_path, indices):
    """Read RGB frames at ``indices``. Shape ``(T, H, W, 3)``, uint8."""
    indices = list(indices)
    if not indices:
        return np.zeros((0, 480, 640, 3), dtype=np.uint8)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open RGB video: {video_path}")
    start, stop = indices[0], indices[-1]
    wanted = set(indices)
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for index in range(start, stop + 1):
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != len(indices):
        raise RuntimeError(
            f"Read {len(frames)} RGB frames from {video_path}, expected {len(indices)}"
        )
    return np.stack(frames)


def thermal_to_celsius(frames):
    """Map stored thermal PNGs onto 15..50 °C."""
    scaled = frames.astype(np.float32) / 255.0
    return scaled * (THERMAL_MAX_C - THERMAL_MIN_C) + THERMAL_MIN_C


def _metadata_row(metadata_csv, user_id, activity, timestamp):
    with open(metadata_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["user_id"] != user_id or row["activity"] != activity:
                continue
            if timestamp is None or row.get("new_timestamp") == timestamp:
                return row
    raise FileNotFoundError(
        f"No metadata row for user {user_id}, activity {activity}, timestamp {timestamp}"
    )


def _slice_recording(recording, start, end):
    clip = {
        "user_id": recording["user_id"],
        "activity": recording["activity"],
        "timestamp": recording["timestamp"],
        "start": start,
        "end": end,
    }
    for key, value in recording.items():
        if key in ("user_id", "activity", "timestamp", "segments", "missing"):
            continue
        if key == "imu":
            clip["imu"] = _slice_frames(value, start, end)
        elif key == "mocap":
            clip["mocap"] = _slice_frames(value, start, end)
        elif key == "depthCamera":
            clip["depthCamera"] = {
                node: _slice_camera(stream, start, end) for node, stream in value.items()
            }
        elif key == "acoustic":
            clip["acoustic"] = {
                node: _slice_acoustic(stream, start, end) for node, stream in value.items()
            }
        else:
            clip[key] = {node: _slice_frames(stream, start, end) for node, stream in value.items()}
    return clip


def _slice_frames(stream, start, end):
    index = _time_index(stream["timestamps"], start, end)
    sliced = {"timestamps": [stream["timestamps"][i] for i in index]}
    for key, value in stream.items():
        if key == "timestamps":
            continue
        if isinstance(value, list):
            sliced[key] = [value[i] for i in index]
        else:
            sliced[key] = value[index]
    return sliced


def _slice_camera(stream, start, end):
    index = _time_index(stream["timestamps"], start, end)
    paths = [stream["depth_paths"][i] for i in index]
    return {
        "timestamps": [stream["timestamps"][i] for i in index],
        "depth": load_depth_frames(paths),
    }


def _slice_acoustic(stream, start, end):
    rate = stream["sample_rate"]
    origin = stream["start_time"]
    first = max(0, int(round((start - origin).total_seconds() * rate)))
    last = max(first, int(round((end - origin).total_seconds() * rate)))
    last = min(last, len(stream["waveform"]))
    return {
        "start_time": origin,
        "sample_rate": rate,
        "waveform": stream["waveform"][first:last],
    }


def _time_index(timestamps, start, end):
    return [i for i, stamp in enumerate(timestamps) if start <= stamp < end]


def _load_mocap(csv_path):
    pose = convert_mocap_csv(csv_path)
    return {
        "timestamps": [_parse_time(stamp) for stamp in pose["timestamps"]],
        "positions": pose["positions"],
    }


def _load_imu(path):
    entries = _read_pickle_stream(path)
    timestamps, frames = [], []
    for entry in entries:
        timestamps.extend(_as_datetime(stamp) for stamp in entry["timestamps"])
        frames.append(np.asarray(entry["data"]))
    data = frames[0] if len(frames) == 1 else np.concatenate(frames, axis=0)
    return {"timestamps": timestamps, "frames": data}


def _load_table(path, time_key, data_key):
    """Stack either one ``(T, ...)`` blob or one record per frame."""
    entries = _read_pickle_stream(path)
    if len(entries) == 1 and isinstance(entries[0].get(time_key), (list, np.ndarray)):
        timestamps = [_as_datetime(stamp) for stamp in np.ravel(entries[0][time_key])]
        frames = np.asarray(entries[0][data_key])
        return {"timestamps": timestamps, "frames": frames}

    timestamps, frames = [], []
    for entry in entries:
        timestamps.append(_as_datetime(entry[time_key]))
        frames.append(np.asarray(entry[data_key]))
    return {"timestamps": timestamps, "frames": np.stack(frames)}


def _load_ira(path):
    stream = _load_table(path, "timestamp", "Detected_Temperature")
    return stream


def _load_wifi(path):
    return _load_table(path, "timestamp", "data")


def _load_uwb(path):
    stream = _load_table(path, "timestamp", "frame")
    return stream


def _load_polar(path):
    stream = _load_table(path, "timestamp", "data")
    stream["frames"] = np.asarray(stream["frames"])
    return stream


def _load_vayyar(path):
    stream = _load_table(path, "timestamps", "data")
    frames = stream["frames"]
    if frames.ndim == 3 and frames.shape[0] == 400 and frames.shape[1] == 100:
        frames = np.transpose(frames, (2, 0, 1))
    stream["frames"] = frames
    return stream


def _load_mmwave(path):
    entries = _read_pickle_stream(path)
    timestamps, points = [], []
    for entry in entries:
        timestamps.append(_as_datetime(entry["timestamp"]))
        frame = entry["data"]
        if all(key in frame for key in ("x", "y", "z", "velocity")):
            points.append(np.stack([frame["x"], frame["y"], frame["z"], frame["velocity"]], axis=-1))
        else:
            points.append(np.zeros((0, 4), dtype=np.float32))
    return {"timestamps": timestamps, "points": points}


def _load_tof(path):
    entries = _read_pickle_stream(path)
    timestamps, bins, depth = [], [], []
    for entry in entries:
        timestamps.append(_as_datetime(entry["timestamp"]))
        bins.append(np.asarray(entry["tof_bins"]))
        depth.append(np.asarray(entry["tof_depth"]))
    bin_array = np.stack(bins).astype(np.float32)
    if bin_array.shape[1:] == (64, 18):
        bin_array = bin_array.reshape(bin_array.shape[0], 8, 8, 18)
    return {
        "timestamps": timestamps,
        "tof_bins": bin_array,
        "tof_depth": np.stack(depth),
    }


def _load_seekthermal(directory):
    stamps_and_paths = _timestamped_images(directory, "thermal_")
    timestamps = [stamp for stamp, _ in stamps_and_paths]
    frames = np.stack([_read_gray(path) for _, path in stamps_and_paths])
    return {"timestamps": timestamps, "frames": frames}


def _load_depth_camera(directory):
    stamps_and_paths = _timestamped_images(directory, "depth_")
    return {
        "timestamps": [stamp for stamp, _ in stamps_and_paths],
        "depth_paths": [path for _, path in stamps_and_paths],
    }


def _load_acoustic(directory):
    wav_path = next(directory.glob("*.wav"))
    log_path = wav_path.with_suffix(".log")
    if not log_path.exists():
        log_path = next(directory.glob("*.log"))
    waveform, sample_rate = _read_wav(wav_path)
    return {
        "start_time": _acoustic_start(log_path),
        "sample_rate": sample_rate,
        "waveform": waveform,
    }


def _timestamped_images(directory, prefix):
    found = []
    for path in directory.glob("*.png"):
        if not path.name.startswith(prefix):
            continue
        stamp = path.name[len(prefix):-4]
        found.append((_parse_time(stamp, "%Y-%m-%d %H:%M:%S.%f"), path))
    if not found:
        raise FileNotFoundError(f"No {prefix}*.png images in {directory}")
    found.sort(key=lambda item: item[0])
    return found


def _acoustic_start(log_path):
    for line in log_path.read_text(errors="replace").splitlines():
        match = _NTP_START.search(line)
        if match:
            return _parse_time(match.group(1), "%Y-%m-%d %H:%M:%S.%f")
        if "Start Recording" in line and "NTP Time:" not in line:
            return _parse_time(line.split("|", 1)[0].strip(), "%Y-%m-%d %H:%M:%S.%f")
    raise ValueError(f"No recording start time in {log_path}")


def _read_wav(path):
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(path))
    mono = waveform.mean(dim=0).numpy().astype(np.float32)
    return mono, int(sample_rate)


def _read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _read_pickle_stream(path):
    import pickle

    records = []
    with open(path, "rb") as handle:
        while True:
            try:
                records.append(pickle.load(handle))
            except EOFError:
                break
    if not records:
        raise ValueError(f"Empty pickle stream: {path}")
    return records


def _as_datetime(value):
    if isinstance(value, datetime):
        return value
    if isinstance(value, np.datetime64):
        return value.astype("datetime64[us]").item()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()
    if isinstance(value, str):
        return _parse_time(value)
    raise TypeError(f"Unsupported timestamp type: {type(value)}")


def _parse_time(text, fmt=None):
    if isinstance(text, datetime):
        return text
    if fmt is not None:
        return datetime.strptime(text, fmt)
    for candidate in (
        "%Y-%m-%d %H.%M.%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H.%M.%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, candidate)
        except ValueError:
            continue
    return datetime.fromisoformat(text)


_LOADERS = {
    "IRA": _load_ira,
    "wifi": _load_wifi,
    "uwb": _load_uwb,
    "polar": _load_polar,
    "vayyar": _load_vayyar,
    "mmWave": _load_mmwave,
    "ToF": _load_tof,
    "seekThermal": _load_seekthermal,
    "depthCamera": _load_depth_camera,
    "acoustic": _load_acoustic,
}


if __name__ == "__main__":
    recording = load_recording()
    print(
        f"user {recording['user_id']} {recording['activity']} {recording['timestamp']}: "
        f"{len(recording['segments'])} segments, missing {recording['missing'] or 'none'}"
    )
    for key, value in recording.items():
        if not isinstance(value, dict):
            continue
        if "timestamps" in value:
            array_key = next(name for name in value if name != "timestamps")
            print(f"  {key}: {len(value['timestamps'])} frames, {array_key} {np.shape(value[array_key])}")
            continue
        for node, stream in value.items():
            if "frames" in stream:
                print(f"  {key} node {node}: {np.shape(stream['frames'])} {stream['frames'].dtype}")
            elif "points" in stream:
                print(f"  {key} node {node}: {len(stream['points'])} point clouds")
            elif "depth_paths" in stream:
                print(f"  {key} node {node}: {len(stream['depth_paths'])} depth frames")
            elif "waveform" in stream:
                print(f"  {key} node {node}: {stream['waveform'].shape} at {stream['sample_rate']} Hz")
            elif "tof_bins" in stream:
                print(f"  {key} node {node}: bins {stream['tof_bins'].shape}")
