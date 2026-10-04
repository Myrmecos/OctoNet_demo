"""Build activity-classification samples from the octonet_entire recording tree.

Sensor bytes are read from the linked dataset and are never written back.
Zip archives (depth, Seek Thermal, acoustic) are decoded in memory. Vayyar
pickles live under ``vayyar_pickle/`` rather than the path stored in the CSV.
"""

import csv
import io
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader_new import (  # noqa: E402
    DEPTH_MAX_METERS,
    _acoustic_start,
    _load_imu,
    _load_ira,
    _load_mmwave,
    _load_mocap,
    _load_polar,
    _load_tof,
    _load_uwb,
    _load_vayyar,
    _load_wifi,
    _parse_time,
    _slice_acoustic,
    _slice_frames,
    segment_bounds,
    thermal_to_celsius,
)

from models.features import (  # noqa: E402
    SHAPES,
    from_acoustic,
    from_images,
    from_imu,
    from_ira,
    from_mmwave,
    from_mocap,
    from_polar,
    from_tof,
    from_uwb,
    from_vayyar,
    from_wifi,
)

# Column in cut_manual.csv, and which node that file belongs to.
MODALITY_COLUMNS = {
    "IRA": "node_1_IRA_data_path",
    "wifi": "node_1_wifi_data_path",
    "uwb": "node_1_uwb_data_path",
    "mmWave": "node_1_mmWave_data_path",
    "ToF": "node_4_ToF_data_path",
    "polar": "node_1_polar_data_path",
    "vayyar": "node_1_vayyar_data_path",
    "seekThermal": "node_1_seekThermal_data_path",
    "depthCamera": "node_1_depthCamera_data_path",
    "acoustic": "node_1_acoustic_data_path",
    "imu": "imu_data_path",
    "mocap": "mocap_data_path",
}

MODALITIES = list(MODALITY_COLUMNS)
CACHE_VERSION = "v2"


def dataset_root():
    """Resolve the octonet_entire link without creating or editing it.

    The link is stored as a relative path (``disk/banana/...``). When that
    relative target is missing, the same path under the filesystem root is used.
    """
    link = ROOT / "octonet_entire"
    if link.is_dir():
        return link.resolve()
    target = Path(link.readlink()) if link.is_symlink() else None
    if target is not None:
        candidate = Path("/") / target
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"octonet_entire does not point at a directory: {link}")


def resolve_source(root, relative):
    """Find a CSV path inside the linked dataset. Returns None when it is absent."""
    if not relative:
        return None
    direct = root / relative
    if direct.exists():
        return direct
    zipped = Path(str(direct) + ".zip")
    if zipped.is_file():
        return zipped
    vayyar = root / "vayyar_pickle" / Path(relative).name
    if vayyar.is_file():
        return vayyar
    return None


def load_metadata(metadata_csv=None, users=None):
    metadata_csv = Path(metadata_csv or ROOT / "cut_manual.csv")
    allowed = None if not users else {str(user) for user in users}
    rows = []
    with open(metadata_csv, newline="") as handle:
        for row in csv.DictReader(handle):
            if allowed is not None and row["user_id"] not in allowed:
                continue
            rows.append(row)
    rows.sort(key=lambda row: (row["user_id"].zfill(6), row["activity"], row.get("new_timestamp", "")))
    return rows


def limit_rows(rows, max_recordings, seed):
    if not max_recordings or max_recordings >= len(rows):
        return list(rows)
    order = np.random.RandomState(seed).permutation(len(rows))
    return [rows[int(index)] for index in order[:max_recordings]]


def recording_key(row):
    return f"{row['user_id']}|{row['activity']}|{row.get('new_timestamp', '')}"


def split_recordings(rows, seed, train_ratio=0.7, val_ratio=0.1):
    """Stratify whole recordings so segments from one take stay in one split."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["activity"], []).append(row)
    train, val, test = [], [], []
    rng = np.random.RandomState(seed)
    for activity in sorted(grouped):
        items = list(grouped[activity])
        rng.shuffle(items)
        count = len(items)
        if count == 1:
            train.extend(items)
            continue
        n_test = max(1, int(round(count * (1.0 - train_ratio - val_ratio))))
        n_val = max(1, int(round(count * val_ratio))) if count - n_test >= 2 else 0
        if n_test + n_val >= count:
            n_test = 1
            n_val = 1 if count > 2 else 0
        test.extend(items[:n_test])
        val.extend(items[n_test:n_test + n_val])
        train.extend(items[n_test + n_val:])
    return {"train": train, "val": val, "test": test}


def _cache_path(modality, row):
    safe = recording_key(row).replace("|", "_")
    return ROOT / "models" / "cache" / CACHE_VERSION / modality / f"{safe}.npz"


def _read_cache(path):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as blob:
            if str(blob["version"]) != CACHE_VERSION:
                return None
            data = np.asarray(blob["data"], dtype=np.float32)
    except (OSError, ValueError, KeyError):
        return None
    return data


def _write_cache(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    stacked = np.zeros((0,), dtype=np.float32) if len(data) == 0 else np.stack(data).astype(np.float32)
    np.savez_compressed(path, version=np.array(CACHE_VERSION), data=stacked)


def collect_split(modality, rows, root=None):
    """Load every middle-cut window of ``modality`` for ``rows``.

    Returns a float32 array of shape (N, C, H, W) and a list of activity names.
    """
    root = dataset_root() if root is None else Path(root)
    features, labels = [], []
    failures = 0
    for index, row in enumerate(rows, start=1):
        try:
            windows = windows_for_row(modality, row, root)
        except Exception as exc:
            failures += 1
            print(f"  skip {recording_key(row)}: {exc}")
            windows = []
        for window in windows:
            features.append(window)
            labels.append(row["activity"])
        if index % 25 == 0 or index == len(rows):
            print(f"  {modality}: {index}/{len(rows)} recordings, {len(features)} windows")
    if failures:
        print(f"  {modality}: {failures} recordings failed to load")
    if not features:
        return np.zeros((0,), dtype=np.float32), []
    return np.stack(features).astype(np.float32), labels


def windows_for_row(modality, row, root):
    cached = _read_cache(_cache_path(modality, row))
    if cached is not None:
        if cached.ndim == 1:
            return []
        return [cached[i] for i in range(cached.shape[0])]
    source = resolve_source(root, row.get(MODALITY_COLUMNS[modality], ""))
    bounds = segment_bounds(row["cut_timestamps"])
    if source is None or not bounds:
        _write_cache(_cache_path(modality, row), [])
        return []
    windows = _windows_from_source(modality, source, bounds)
    _write_cache(_cache_path(modality, row), windows)
    return windows


def _windows_from_source(modality, source, bounds):
    if modality in ("seekThermal", "depthCamera"):
        return _image_windows(modality, source, bounds)
    if modality == "acoustic":
        return _acoustic_windows(source, bounds)
    stream = _LOADERS[modality](source)
    windows = []
    for start, end in bounds:
        sliced = _slice_frames(stream, start, end)
        try:
            sample = _TABLE_FEATURES[modality](sliced)
        except ValueError:
            continue
        windows.append(np.ascontiguousarray(sample, dtype=np.float32))
    return windows


def _image_windows(modality, source, bounds):
    if source.is_dir():
        return _image_windows_from_directory(modality, source, bounds)
    prefix = "thermal_" if modality == "seekThermal" else "depth_"
    frame_count = SHAPES[modality][0]
    with zipfile.ZipFile(source) as archive:
        frames = []
        for name in archive.namelist():
            base = Path(name).name
            if not base.startswith(prefix) or not base.endswith(".png"):
                continue
            stamp = _parse_time(base[len(prefix):-4], "%Y-%m-%d %H:%M:%S.%f")
            frames.append((stamp, name))
        frames.sort(key=lambda item: item[0])
        windows = []
        for start, end in bounds:
            names = [name for stamp, name in frames if start <= stamp < end]
            if not names:
                continue
            if len(names) > frame_count:
                picked = np.linspace(0, len(names) - 1, frame_count).round().astype(int)
                names = [names[int(i)] for i in picked]
            decoded = []
            for name in names:
                try:
                    decoded.append(_decode_gray(archive.read(name)))
                except FileNotFoundError:
                    continue
            if not decoded:
                continue
            stack = np.stack(decoded)
            if modality == "seekThermal":
                stack = thermal_to_celsius(stack)
            else:
                stack = stack.astype(np.float32) / 255.0 * DEPTH_MAX_METERS
            windows.append(np.ascontiguousarray(from_images(stack, modality), dtype=np.float32))
    return windows


def _decode_gray(payload):
    import cv2

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError("could not decode an image from the zip archive")
    if image.ndim == 3:
        image = image[:, :, 0]
    return image


def _image_windows_from_directory(modality, source, bounds):
    from data_loader_new import _load_depth_camera, _load_seekthermal, load_depth_frames

    if modality == "seekThermal":
        stream = _load_seekthermal(source)
        windows = []
        for start, end in bounds:
            sliced = _slice_frames(stream, start, end)
            if len(sliced["frames"]) == 0:
                continue
            celsius = thermal_to_celsius(sliced["frames"])
            windows.append(np.ascontiguousarray(from_images(celsius, modality), dtype=np.float32))
        return windows
    stream = _load_depth_camera(source)
    windows = []
    for start, end in bounds:
        sliced = _slice_frames({"timestamps": stream["timestamps"], "frames": stream["depth_paths"]}, start, end)
        paths = sliced["frames"]
        if len(paths) == 0:
            continue
        frame_count = SHAPES[modality][0]
        if len(paths) > frame_count:
            picked = np.linspace(0, len(paths) - 1, frame_count).round().astype(int)
            paths = [paths[int(i)] for i in picked]
        depth = load_depth_frames(paths)
        windows.append(np.ascontiguousarray(from_images(depth, modality), dtype=np.float32))
    return windows


def _acoustic_windows(source, bounds):
    import torch
    import torchaudio

    if source.is_dir():
        from data_loader_new import _load_acoustic

        stream = _load_acoustic(source)
    else:
        with zipfile.ZipFile(source) as archive:
            wav_name, log_name = _acoustic_members(archive)
            waveform, sample_rate = torchaudio.load(io.BytesIO(archive.read(wav_name)))
            log_path = ROOT / "models" / "cache" / "_acoustic_start.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_bytes(archive.read(log_name))
            start_time = _acoustic_start(log_path)
        stream = {
            "start_time": start_time,
            "sample_rate": int(sample_rate),
            "waveform": waveform.mean(dim=0).numpy().astype(np.float32),
        }
    windows = []
    for start, end in bounds:
        clip = _slice_acoustic(stream, start, end)
        try:
            mel = _log_mel(clip["waveform"], clip["sample_rate"], torch, torchaudio)
            sample = from_acoustic(mel)
        except ValueError:
            continue
        windows.append(np.ascontiguousarray(sample, dtype=np.float32))
    return windows


def _acoustic_members(archive):
    wavs = [name for name in archive.namelist() if name.endswith(".wav") and archive.getinfo(name).file_size > 0]
    if not wavs:
        raise FileNotFoundError("acoustic zip has no wav")
    wavs.sort(key=lambda name: (-archive.getinfo(name).file_size, name.count("/")))
    wav_name = wavs[0]
    parent = str(Path(wav_name).parent)
    logs = [
        name for name in archive.namelist()
        if name.endswith(".log") and str(Path(name).parent) == parent and archive.getinfo(name).file_size > 0
    ]
    if not logs:
        logs = [name for name in archive.namelist() if name.endswith(".log") and archive.getinfo(name).file_size > 0]
    if not logs:
        raise FileNotFoundError("acoustic zip has no log")
    return wav_name, logs[0]


def _log_mel(waveform, sample_rate, torch, torchaudio):
    wave = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
    if sample_rate != 16000:
        wave = torchaudio.transforms.Resample(sample_rate, 16000)(wave)
    if wave.shape[-1] < 1024:
        raise ValueError("acoustic clip is shorter than one FFT window")
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
    )(wave)
    return torch.log1p(mel).squeeze(0).numpy()


def _ira_feature(sliced):
    return from_ira(sliced["frames"])


def _wifi_feature(sliced):
    return from_wifi(sliced["frames"])


def _uwb_feature(sliced):
    return from_uwb(sliced["frames"])


def _mmwave_feature(sliced):
    return from_mmwave(sliced["points"])


def _tof_feature(sliced):
    return from_tof(sliced["tof_bins"])


def _polar_feature(sliced):
    return from_polar(sliced["frames"])


def _vayyar_feature(sliced):
    return from_vayyar(sliced["frames"])


def _imu_feature(sliced):
    return from_imu(sliced["frames"])


def _mocap_feature(sliced):
    return from_mocap(sliced["positions"])


def _load_mocap_capture(path):
    """Load a mocap CSV, accepting Motive's Chinese 上午/下午 meridian."""
    import mocap_convert

    original = mocap_convert._parse_capture_start

    def parse(text):
        translated = str(text).replace("上午", "AM").replace("下午", "PM")
        return original(translated)

    mocap_convert._parse_capture_start = parse
    try:
        return _load_mocap(path)
    finally:
        mocap_convert._parse_capture_start = original


_LOADERS = {
    "IRA": _load_ira,
    "wifi": _load_wifi,
    "uwb": _load_uwb,
    "mmWave": _load_mmwave,
    "ToF": _load_tof,
    "polar": _load_polar,
    "vayyar": _load_vayyar,
    "imu": _load_imu,
    "mocap": _load_mocap_capture,
}

_TABLE_FEATURES = {
    "IRA": _ira_feature,
    "wifi": _wifi_feature,
    "uwb": _uwb_feature,
    "mmWave": _mmwave_feature,
    "ToF": _tof_feature,
    "polar": _polar_feature,
    "vayyar": _vayyar_feature,
    "imu": _imu_feature,
    "mocap": _mocap_feature,
}
