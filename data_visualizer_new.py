"""Write the OctoNet demo views for one prepared recording.

Each view is an mp4 under ``viz_output/``. The timeline is node-1 depth. Other
streams use ``i / (T_depth - 1) * (T_src - 1)``. RGB video is not read.

Example:
    python data_visualizer_new.py --segment 0 --max-frames 48
"""

import argparse
import os
import traceback
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import matplotlib

try:
    get_ipython()
except NameError:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_loader_new import ROOT, _slice_recording, load_recording, thermal_to_celsius

PANEL_W = 320
PANEL_H = 240
DEFAULT_OUTPUT = ROOT / "viz_output"

# Same bones as OctoNet/dataset_visualizer.py.
MOCAP_BONES = (
    (0, 1), (1, 2), (2, 3),
    (4, 5), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11),
    (12, 13), (13, 14), (14, 15),
    (16, 17), (17, 18), (18, 19),
    (0, 16), (0, 12),
)

# 40 MHz WiFi guard and null subcarriers removed by the original WiFi view.
WIFI_DROP_40MHZ = {5, 33, 47, 66, 80, 108}

IMU_GROUPS = (
    ("acc", (0, 1, 2)),
    ("mag", (3, 4, 5)),
    ("euler", (6, 7, 8)),
    ("quat", (9, 10, 11, 12)),
)


def map_index(index, source_count, target_count):
    """Map frame ``index`` in a stream of length ``source_count`` onto ``target_count``."""
    if source_count <= 1 or target_count <= 1:
        return 0
    mapped = int(index / (source_count - 1) * (target_count - 1))
    return min(mapped, target_count - 1)


def sample_indices(count, max_frames):
    """Frame indices along a camera stream. ``max_frames <= 0`` keeps every frame."""
    if count <= 0:
        return []
    if max_frames is None or max_frames <= 0 or count <= max_frames:
        return list(range(count))
    chosen = np.rint(np.linspace(0, count - 1, max_frames)).astype(int)
    return np.unique(chosen).tolist()


def reference_length(clip):
    """Number of frames on the alignment clock. Node-1 depth is preferred."""
    cameras = clip.get("depthCamera") or {}
    if cameras:
        node = 1 if 1 in cameras else next(iter(sorted(cameras)))
        depth = cameras[node].get("depth")
        if depth is not None and len(depth):
            return len(depth)
    for value in clip.values():
        if isinstance(value, dict) and value.get("timestamps"):
            return len(value["timestamps"])
        if not isinstance(value, dict):
            continue
        for stream in value.values():
            if isinstance(stream, dict) and stream.get("timestamps"):
                return len(stream["timestamps"])
    raise ValueError("The clip has no frames to align against")


def _only(clip, modality):
    """The one loaded node for ``modality``."""
    streams = clip.get(modality) or {}
    if not streams:
        raise KeyError(modality)
    node = 1 if 1 in streams else next(iter(sorted(streams)))
    return node, streams[node]


def fit(image, width=PANEL_W, height=PANEL_H):
    shrinking = image.shape[1] > width or image.shape[0] > height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_NEAREST
    return cv2.resize(image, (width, height), interpolation=interpolation)


def label(image, text):
    """Burn a name into the corner. Skipped for notebook stills; the figure title is enough."""
    if not label.enabled:
        return image
    panel = image.copy()
    origin = (8, 22)
    cv2.putText(panel, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(panel, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return panel


label.enabled = True


def colorize(image2d, vmin=None, vmax=None, cmap=cv2.COLORMAP_VIRIDIS):
    data = np.asarray(image2d, dtype=np.float32)
    finite = np.isfinite(data)
    if vmin is None:
        vmin = float(data[finite].min()) if finite.any() else 0.0
    if vmax is None:
        vmax = float(data[finite].max()) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-6
    norm = np.zeros(data.shape, dtype=np.float32)
    if finite.any():
        norm[finite] = (data[finite] - vmin) / (vmax - vmin)
    gray = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cmap)


def heatmap_limits(matrix):
    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.percentile(finite, (2, 98))
    return float(low), float(high)


def prepare_heatmap(matrix):
    low, high = heatmap_limits(np.asarray(matrix, dtype=np.float32))
    image = colorize(matrix, low, high, cv2.COLORMAP_MAGMA)
    return cv2.resize(image, (PANEL_W, PANEL_H), interpolation=cv2.INTER_NEAREST)


def cursor_on(image, cursor, count):
    """Draw the time cursor. ``cursor_on.enabled`` is false for a still key frame."""
    panel = image.copy()
    if not cursor_on.enabled:
        return panel
    x = int(round(cursor / max(count - 1, 1) * (image.shape[1] - 1)))
    cv2.line(panel, (x, 0), (x, image.shape[0] - 1), (255, 255, 255), 1)
    return panel


cursor_on.enabled = True


def grid(panels, columns):
    tiles = list(panels)
    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank)
    rows = [np.hstack(tiles[start:start + columns]) for start in range(0, len(tiles), columns)]
    return np.vstack(rows)


def write_mp4(path, frames, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    width -= width % 2
    height -= height % 2
    written = []
    for frame in frames:
        image = frame
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height))
        written.append(np.ascontiguousarray(image))
    writer = None
    for codec in ("mp4v", "avc1", "XVID"):
        candidate = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height)
        )
        if candidate.isOpened():
            writer = candidate
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"Could not open a video writer for {path}")
    for frame in written:
        writer.write(frame)
    writer.release()
    return path


def view_depth(clip, ref_count, indices):
    """Depth in meters for the one loaded camera node."""
    node, stream = _only(clip, "depthCamera")
    frames = []
    for index in indices:
        depth_at = map_index(index, ref_count, len(stream["depth"]))
        frames.append(label(fit(colorize(stream["depth"][depth_at])), f"depth {node} m"))
    return frames


def view_seek(clip, ref_count, indices):
    """Seek Thermal, in Celsius, for the one loaded node."""
    node, stream = _only(clip, "seekThermal")
    frames = []
    for index in indices:
        thermal_at = map_index(index, ref_count, len(stream["frames"]))
        celsius = thermal_to_celsius(stream["frames"][thermal_at])
        frames.append(label(fit(colorize(celsius, 15, 50, cv2.COLORMAP_INFERNO)), f"thermal {node}"))
    return frames


def _fill_ira(frame):
    """Fill the unused checkerboard pixels in an MLX90640 frame for display."""
    data = np.asarray(frame, dtype=np.float32).copy()
    missing = data == 0
    if not missing.any():
        return data
    padded = np.pad(data, 1, mode="edge")
    total = np.zeros_like(data)
    weight = np.zeros_like(data)
    for y_shift, x_shift in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        shifted = padded[
            1 + y_shift:1 + y_shift + data.shape[0],
            1 + x_shift:1 + x_shift + data.shape[1],
        ]
        valid = shifted != 0
        total += np.where(valid, shifted, 0)
        weight += valid
    can_fill = missing & (weight > 0)
    data[can_fill] = total[can_fill] / weight[can_fill]
    return data


def view_ira(clip, ref_count, indices):
    """IRA temperature map for the one loaded node."""
    node, stream = _only(clip, "IRA")
    frames = []
    for index in indices:
        thermal_at = map_index(index, ref_count, len(stream["frames"]))
        filled = _fill_ira(stream["frames"][thermal_at])
        frames.append(label(fit(colorize(filled, cmap=cv2.COLORMAP_INFERNO)), f"IRA {node}"))
    return frames


def _wifi_amplitude(frames):
    keep = [bin_index for bin_index in range(frames.shape[-1]) if bin_index not in WIFI_DROP_40MHZ]
    amplitude = np.abs(frames[:, :, keep])
    return amplitude.reshape(len(frames), -1).T


def view_wifi(clip, ref_count, indices):
    """WiFi amplitude for the one loaded node. The 40 MHz null subcarriers are removed."""
    node, stream = _only(clip, "wifi")
    amplitude = _wifi_amplitude(stream["frames"])
    base = prepare_heatmap(amplitude)
    frames = []
    for index in indices:
        cursor = map_index(index, ref_count, amplitude.shape[1])
        frames.append(label(cursor_on(base, cursor, amplitude.shape[1]), f"wifi {node}"))
    return frames


def view_tof(clip, ref_count, indices):
    """ToF, averaged over the 18 range bins, for the one loaded node."""
    node, stream = _only(clip, "ToF")
    bins = stream["tof_bins"]
    frames = []
    for index in indices:
        tof_at = map_index(index, ref_count, len(bins))
        image = np.mean(bins[tof_at], axis=-1)
        frames.append(label(fit(colorize(image)), f"ToF {node}"))
    return frames


def _gather_points(points, index):
    chunks = []
    for frame_index in range(index, min(index + 3, len(points))):
        cloud = np.asarray(points[frame_index], dtype=np.float32)
        if cloud.ndim == 2 and cloud.shape[0] > 0 and cloud.shape[1] >= 2:
            chunks.append(cloud)
    if not chunks:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def _valid_points(points):
    """Drop empty slots and non-finite or absurd radar returns."""
    cloud = np.asarray(points, dtype=np.float32)
    if cloud.ndim != 2 or cloud.shape[0] == 0 or cloud.shape[1] < 3:
        return np.zeros((0, 4), dtype=np.float32)
    keep = np.isfinite(cloud).all(axis=1)
    coords = cloud[:, :3]
    keep &= (np.abs(coords) < 20).all(axis=1)
    keep &= ~(np.abs(coords) < 1e-6).all(axis=1)
    return cloud[keep]


def _xz_limits(points):
    """Stable x (lateral) and z (height) limits for one node's clouds."""
    chunks = []
    for cloud in points:
        valid = _valid_points(cloud)
        if len(valid):
            chunks.append(valid[:, :3])
    if not chunks:
        return (-1.0, 1.0), (-1.0, 1.0)
    coords = np.concatenate(chunks, axis=0)

    def span(column):
        low, high = np.percentile(column, (1, 99))
        if high - low < 1.0:
            center = (high + low) / 2.0
            return center - 0.5, center + 0.5
        margin = 0.1 * (high - low)
        return float(low - margin), float(high + margin)

    return span(coords[:, 0]), span(coords[:, 2])


def _scatter_panel(points, title, x_limits, z_limits):
    """x-z cloud. Color is forward range y. Limits stay fixed for the whole cut."""
    canvas = np.full((PANEL_H, PANEL_W, 3), 32, np.uint8)
    cloud = _valid_points(points)
    if len(cloud):
        x_low, x_high = x_limits
        z_low, z_high = z_limits
        inside = (
            (cloud[:, 0] >= x_low) & (cloud[:, 0] <= x_high)
            & (cloud[:, 2] >= z_low) & (cloud[:, 2] <= z_high)
        )
        cloud = cloud[inside]
    if len(cloud):
        x_low, x_high = x_limits
        z_low, z_high = z_limits
        px = ((cloud[:, 0] - x_low) / (x_high - x_low) * (PANEL_W - 1)).astype(int)
        py = ((z_high - cloud[:, 2]) / (z_high - z_low) * (PANEL_H - 1)).astype(int)
        distance = cloud[:, 1]
        span = max(float(distance.max() - distance.min()), 1e-3)
        shade = (distance - float(distance.min())) / span
        for x, y, tone in zip(px, py, shade):
            color = (int(255 * (1.0 - tone)), int(180 * tone + 40), int(255 * tone))
            cv2.circle(canvas, (int(x), int(y)), 2, color, -1)
    return label(canvas, title)


def view_mmwave(clip, ref_count, indices):
    """mmWave point cloud for the one loaded node. Three frames are merged."""
    node, stream = _only(clip, "mmWave")
    points = stream["points"]
    x_limits, z_limits = _xz_limits(points)
    frames = []
    for index in indices:
        cloud_at = map_index(index, ref_count, len(points))
        frames.append(_scatter_panel(
            _gather_points(points, cloud_at), f"mmWave {node}", x_limits, z_limits
        ))
    return frames


def view_vayyar(clip, ref_count, indices):
    """Vayyar amplitude, averaged over the 100 ADC steps. Rows are the 400 antennas."""
    _, stream = _only(clip, "vayyar")
    amplitude = np.abs(stream["frames"]).mean(axis=-1).T
    base = prepare_heatmap(amplitude)
    frames = []
    for index in indices:
        cursor = map_index(index, ref_count, amplitude.shape[1])
        frames.append(label(cursor_on(base, cursor, amplitude.shape[1]), "vayyar"))
    return frames


def view_uwb(clip, ref_count, indices):
    """UWB CIR magnitude, range by time."""
    _, stream = _only(clip, "uwb")
    cir = np.abs(stream["frames"])
    base = prepare_heatmap(cir.T)
    frames = []
    for index in indices:
        cursor = map_index(index, ref_count, cir.shape[0])
        frames.append(label(cursor_on(base, cursor, cir.shape[0]), "UWB"))
    return frames


def _log_mel(waveform, sample_rate):
    """16 kHz log-mel used by the original acoustic collate: n_fft 1024, hop 512, 128 mels."""
    import torch
    import torchaudio

    wave = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
    if sample_rate != 16000:
        wave = torchaudio.transforms.Resample(sample_rate, 16000)(wave)
    if wave.shape[-1] < 1024:
        raise ValueError("acoustic clip is shorter than one FFT window")
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128
    )(wave)
    return torch.log1p(mel).squeeze(0).numpy()


def view_acoustic(clip, ref_count, indices):
    """Log-mel spectrogram for the one loaded acoustic node."""
    node, stream = _only(clip, "acoustic")
    mel = _log_mel(stream["waveform"], stream["sample_rate"])
    mel = mel[::-1]  # low frequency at the bottom
    base = prepare_heatmap(mel)
    frames = []
    for index in indices:
        cursor = map_index(index, ref_count, mel.shape[1])
        frames.append(label(cursor_on(base, cursor, mel.shape[1]), f"audio {node}"))
    return frames


def _lines_base(series, height):
    """Draw ``series`` of shape (T, channels, sensors) on a white panel."""
    if series.ndim == 1:
        series = series[:, None, None]
    elif series.ndim == 2:
        series = series[:, :, None]
    count, channels, sensors = series.shape
    finite = series[np.isfinite(series)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    if high <= low:
        high = low + 1.0
    canvas = np.full((height, PANEL_W, 3), 255, np.uint8)
    palette = ((40, 40, 200), (40, 140, 40), (200, 60, 40), (160, 40, 160))
    for channel in range(channels):
        color = palette[channel % len(palette)]
        for sensor in range(sensors):
            column = series[:, channel, sensor]
            points = []
            for time_index, value in enumerate(column):
                x = int(round(time_index / max(count - 1, 1) * (PANEL_W - 1)))
                y = int(round((high - value) / (high - low) * (height - 16) + 8))
                points.append((x, int(np.clip(y, 0, height - 1))))
            cv2.polylines(canvas, [np.asarray(points, np.int32)], False, color, 1, cv2.LINE_AA)
    return canvas


def _line_video(series, ref_count, indices, title, height=PANEL_H):
    base = _lines_base(np.asarray(series, dtype=np.float32), height)
    count = series.shape[0]
    frames = []
    for index in indices:
        cursor = map_index(index, ref_count, count)
        frames.append(label(cursor_on(base, cursor, count), title))
    return frames


def view_imu(clip, ref_count, indices):
    """Four IMU rows: acceleration, magnetic field, Euler angles, quaternion. All 17 markers."""
    imu = clip.get("imu")
    if imu is None:
        raise KeyError("imu")
    frames_in = np.asarray(imu["frames"], dtype=np.float32)
    row_h = 120
    bases = [(name, _lines_base(frames_in[:, list(channels), :], row_h)) for name, channels in IMU_GROUPS]
    frames = []
    count = len(frames_in)
    for index in indices:
        cursor = map_index(index, ref_count, count)
        rows = [label(cursor_on(base, cursor, count), name) for name, base in bases]
        frames.append(np.vstack(rows))
    return frames


def view_polar(clip, ref_count, indices):
    """Polar heart rate, in BPM."""
    _, stream = _only(clip, "polar")
    heart_rate = np.squeeze(stream["frames"])
    return _line_video(np.atleast_1d(heart_rate), ref_count, indices, "polar BPM")


def _display_pose(positions):
    """Display-only rotation: +90 degrees about Z, and Z scaled by 10."""
    shown = np.empty_like(positions, dtype=np.float64)
    shown[..., 0] = -positions[..., 1]
    shown[..., 1] = positions[..., 0]
    shown[..., 2] = positions[..., 2] * 10.0
    return shown


def _fig_bgr(fig):
    fig.canvas.draw()
    canvas = fig.canvas
    if hasattr(canvas, "buffer_rgba"):
        image = cv2.cvtColor(np.asarray(canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    else:
        width, height = canvas.get_width_height()
        raw = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        image = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
    plt.close(fig)
    return image


def view_mocap(clip, ref_count, indices):
    """20-joint skeleton from one viewing angle."""
    mocap = clip.get("mocap")
    if mocap is None:
        raise KeyError("mocap")
    shown = _display_pose(mocap["positions"])
    flat = shown.reshape(-1, 3)
    low = flat.min(axis=0) - 50
    high = flat.max(axis=0) + 50
    frames = []
    for index in indices:
        pose_at = map_index(index, ref_count, len(shown))
        pose = shown[pose_at]
        fig = plt.figure(figsize=(4.2, 3.4), dpi=100)
        axis = fig.add_subplot(111, projection="3d")
        axis.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=8, c="tab:blue")
        for start, end in MOCAP_BONES:
            axis.plot(
                pose[[start, end], 0],
                pose[[start, end], 1],
                pose[[start, end], 2],
                color="tab:orange",
                linewidth=2,
            )
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
        axis.set_zlim(low[2], high[2])
        axis.set_xlabel("x", labelpad=6)
        axis.set_ylabel("y", labelpad=6)
        axis.set_zlabel("z", labelpad=6)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=18, azim=-60)
        if label.enabled:
            axis.set_title("mocap")
        fig.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.95)
        frames.append(fit(_fig_bgr(fig)))
    return frames


VIEWS = {
    "depth": view_depth,
    "seek": view_seek,
    "ira": view_ira,
    "wifi": view_wifi,
    "tof": view_tof,
    "mmwave": view_mmwave,
    "vayyar": view_vayyar,
    "uwb": view_uwb,
    "acoustic": view_acoustic,
    "imu": view_imu,
    "polar": view_polar,
    "mocap": view_mocap,
}


def visualize_clip(clip, output_dir, max_frames=48, fps=10, views=None):
    """Write one mp4 per view for a clip from ``iter_segments``."""
    output_dir = Path(output_dir)
    ref_count = reference_length(clip)
    indices = sample_indices(ref_count, max_frames)
    if not indices:
        raise ValueError("The clip has no frames to align against")
    selected = list(VIEWS if views is None else views)
    unknown = [name for name in selected if name not in VIEWS]
    if unknown:
        known = ", ".join(VIEWS)
        raise ValueError(f"Unknown views: {', '.join(unknown)}. Choose from {known}")
    written = []
    for name in selected:
        out_path = output_dir / f"{name}.mp4"
        try:
            frames = VIEWS[name](clip, ref_count, indices)
            write_mp4(out_path, frames, fps)
        except KeyError as exc:
            print(f"skip {name}: missing {exc}")
            continue
        except Exception:
            traceback.print_exc()
            print(f"failed {name}")
            continue
        written.append(out_path)
        print(f"wrote {out_path}")
    return written


def render_keyframes(clip, frame_index=None, views=None):
    """Return one image per view, aligned to a single node-1 depth frame.

    ``frame_index`` defaults to the middle depth frame of the clip. Time-series
    views draw the whole cut. The mp4 cursor and the burned-in corner label are
    omitted here; the notebook figure title names the modality. RGB is not used.
    """
    ref_count = reference_length(clip)
    if frame_index is None:
        frame_index = ref_count // 2
    frame_index = int(np.clip(frame_index, 0, ref_count - 1))
    selected = list(VIEWS if views is None else views)
    unknown = [name for name in selected if name not in VIEWS]
    if unknown:
        known = ", ".join(VIEWS)
        raise ValueError(f"Unknown views: {', '.join(unknown)}. Choose from {known}")
    previous_cursor = cursor_on.enabled
    previous_label = label.enabled
    cursor_on.enabled = False
    label.enabled = False
    images = {}
    try:
        for name in selected:
            try:
                frames = VIEWS[name](clip, ref_count, [frame_index])
            except KeyError as exc:
                print(f"skip {name}: missing {exc}")
                continue
            images[name] = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
    finally:
        cursor_on.enabled = previous_cursor
        label.enabled = previous_label
    return images


# Notebook stills for these views stay as color images, with no axis names.
_IMAGE_VIEWS = ("depth", "seek", "ira", "tof")

# IMU rows. Channel order matches the loader: acc, mag, Euler, quaternion.
_IMU_ROWS = (
    ("Acceleration (m/s²)", (0, 1, 2), ("x", "y", "z")),
    ("Magnetic field", (3, 4, 5), ("x", "y", "z")),
    ("Euler angle (deg)", (6, 7, 8), ("x", "y", "z")),
    ("Quaternion", (9, 10, 11, 12), ("w", "x", "y", "z")),
)
_COMPONENT_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red")


def _elapsed_seconds(timestamps):
    """Seconds from the first stamp of this stream."""
    origin = timestamps[0]
    return np.asarray([(stamp - origin).total_seconds() for stamp in timestamps], dtype=np.float64)


def _mel_center_hz(n_mels=128, sample_rate=16000, n_fft=1024):
    """Center frequency, in Hz, of each mel bin from ``_log_mel``."""
    import torchaudio

    n_freqs = n_fft // 2 + 1
    banks = torchaudio.functional.melscale_fbanks(
        n_freqs, 0.0, float(sample_rate) / 2.0, n_mels, sample_rate, None, "htk"
    )
    banks = np.asarray(banks, dtype=np.float64)
    if banks.shape[0] == n_mels:
        banks = banks.T
    freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)
    weight = np.maximum(banks.sum(axis=0), 1e-12)
    return (freqs @ banks) / weight


def _plot_wifi(clip, title):
    """One heatmap per transmitter. Rows are the kept subcarrier indices."""
    _, stream = _only(clip, "wifi")
    frames = np.asarray(stream["frames"])
    keep = np.asarray(
        [bin_index for bin_index in range(frames.shape[-1]) if bin_index not in WIFI_DROP_40MHZ]
    )
    amplitude = np.abs(frames[:, :, keep])
    times = _elapsed_seconds(stream["timestamps"])
    low, high = heatmap_limits(amplitude)
    transmitters = amplitude.shape[1]
    fig, axes = plt.subplots(
        transmitters, 1, figsize=(9, 3.2 * transmitters), sharex=True, sharey=True, constrained_layout=True
    )
    axes = np.atleast_1d(axes)
    mesh = None
    for transmitter, ax in enumerate(axes):
        mesh = ax.pcolormesh(
            times,
            keep,
            amplitude[:, transmitter, :].T,
            cmap="magma",
            shading="nearest",
            vmin=low,
            vmax=high,
        )
        ax.set_ylabel("subcarrier")
        ax.set_title(f"transmitter {transmitter + 1}")
        ax.tick_params(axis="both", labelsize=9)
    axes[-1].set_xlabel("time (s)")
    fig.colorbar(mesh, ax=axes.ravel().tolist(), label="amplitude", fraction=0.046, pad=0.02)
    if title:
        fig.suptitle(title)
    return fig


def _plot_mmwave(clip, title):
    """3D cloud at the key frame. The next two frames are merged, as in the video."""
    _, stream = _only(clip, "mmWave")
    points = stream["points"]
    ref_count = reference_length(clip)
    cloud_at = map_index(ref_count // 2, ref_count, len(points))
    cloud = _valid_points(_gather_points(points, cloud_at))
    fig = plt.figure(figsize=(8, 6.5), dpi=140)
    axis = fig.add_subplot(111, projection="3d")
    if len(cloud):
        coords = cloud[:, :3]
        scatter = axis.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            c=cloud[:, 3], s=18, cmap="viridis", depthshade=False,
        )
        fig.colorbar(scatter, ax=axis, label="velocity", fraction=0.046, pad=0.1)
        spans = []
        for column in range(3):
            low, high = np.percentile(coords[:, column], (1, 99))
            if high - low < 0.5:
                mid = (high + low) / 2.0
                low, high = mid - 0.25, mid + 0.25
            else:
                margin = 0.1 * (high - low)
                low, high = low - margin, high + margin
            spans.append((float(low), float(high)))
    else:
        spans = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    axis.set_xlim(*spans[0])
    axis.set_ylim(*spans[1])
    axis.set_zlim(*spans[2])
    axis.set_box_aspect(tuple(high - low for low, high in spans))
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_zlabel("z (m)")
    axis.view_init(elev=22, azim=-60)
    if title:
        axis.set_title(title)
    fig.tight_layout()
    return fig


def _plot_heatmap_time(clip, modality, ylabel, color_label, title, reduce=None):
    """Whole-cut heatmap. ``reduce`` maps frames to a 2D array shaped (time, rows)."""
    _, stream = _only(clip, modality)
    frames = np.asarray(stream["frames"])
    values = frames if reduce is None else reduce(frames)
    values = np.abs(values)
    times = _elapsed_seconds(stream["timestamps"])
    rows = np.arange(values.shape[1])
    low, high = heatmap_limits(values)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(
        times, rows, values.T, cmap="magma", shading="nearest", vmin=low, vmax=high
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="both", labelsize=9)
    fig.colorbar(mesh, ax=ax, label=color_label)
    if title:
        ax.set_title(title)
    return fig


def _plot_vayyar(clip, title):
    """Amplitude averaged over the 100 ADC steps. Rows are the 400 antennas."""
    return _plot_heatmap_time(
        clip, "vayyar", "antenna", "amplitude", title,
        reduce=lambda frames: np.abs(frames).mean(axis=-1),
    )


def _plot_uwb(clip, title):
    """CIR magnitude. Columns of the stored frames are range bins."""
    return _plot_heatmap_time(clip, "uwb", "range bin", "magnitude", title)


def _plot_acoustic(clip, title):
    """Log-mel spectrogram. Frequency is the mel-bin center in Hz."""
    _, stream = _only(clip, "acoustic")
    mel = _log_mel(stream["waveform"], stream["sample_rate"])
    hop = 512
    rate = 16000
    times = np.arange(mel.shape[1]) * (hop / rate)
    freqs = _mel_center_hz(n_mels=mel.shape[0], sample_rate=rate)
    low, high = heatmap_limits(mel)
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    mesh = ax.pcolormesh(times, freqs, mel, cmap="magma", shading="nearest", vmin=low, vmax=high)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.tick_params(axis="both", labelsize=9)
    fig.colorbar(mesh, ax=ax, label="log-mel")
    if title:
        ax.set_title(title)
    return fig


def _plot_imu(clip, title):
    """Four rows. Each component of every marker is drawn, with its own amplitude scale."""
    imu = clip.get("imu")
    if imu is None:
        raise KeyError("imu")
    frames = np.asarray(imu["frames"], dtype=np.float64)
    times = _elapsed_seconds(imu["timestamps"])
    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    for ax, (ylabel, channels, names) in zip(axes, _IMU_ROWS):
        for channel, name, color in zip(channels, names, _COMPONENT_COLORS):
            series = frames[:, channel, :]
            for sensor in range(series.shape[1]):
                ax.plot(
                    times, series[:, sensor], color=color, lw=0.8, alpha=0.55,
                    label=name if sensor == 0 else None,
                )
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right", fontsize=8, ncol=len(channels), framealpha=0.9)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    if title:
        fig.suptitle(title)
    return fig


def _plot_polar(clip, title):
    """Heart rate for the whole cut."""
    _, stream = _only(clip, "polar")
    heart_rate = np.asarray(stream["frames"], dtype=np.float64).reshape(-1)
    times = _elapsed_seconds(stream["timestamps"])
    fig, ax = plt.subplots(figsize=(9, 3.2), constrained_layout=True)
    ax.plot(times, heart_rate, color="tab:red", marker="o", lw=1.5)
    low = float(heart_rate.min())
    high = float(heart_rate.max())
    pad = max(1.0, 0.15 * (high - low if high > low else 1.0))
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlim(float(times[0]), float(times[-1]) if len(times) > 1 else float(times[0]) + 1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heart rate (BPM)")
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(True, alpha=0.3)
    if title:
        ax.set_title(title)
    return fig


def _plot_mocap(clip, title):
    """Skeleton at the key frame, drawn directly so the notebook keeps the figure resolution."""
    mocap = clip.get("mocap")
    if mocap is None:
        raise KeyError("mocap")
    shown = _display_pose(mocap["positions"])
    flat = shown.reshape(-1, 3)
    low = flat.min(axis=0) - 50
    high = flat.max(axis=0) + 50
    ref_count = reference_length(clip)
    pose = shown[map_index(ref_count // 2, ref_count, len(shown))]
    fig = plt.figure(figsize=(8, 7), dpi=160)
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(pose[:, 0], pose[:, 1], pose[:, 2], s=28, c="tab:blue", depthshade=False)
    for start, end in MOCAP_BONES:
        axis.plot(
            pose[[start, end], 0],
            pose[[start, end], 1],
            pose[[start, end], 2],
            color="tab:orange",
            linewidth=2.5,
        )
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_zlim(low[2], high[2])
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=18, azim=-60)
    axis.tick_params(axis="both", labelsize=8)
    if title:
        axis.set_title(title)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.92)
    return fig


_NOTEBOOK_PLOTS = {
    "wifi": _plot_wifi,
    "mmwave": _plot_mmwave,
    "vayyar": _plot_vayyar,
    "uwb": _plot_uwb,
    "acoustic": _plot_acoustic,
    "imu": _plot_imu,
    "polar": _plot_polar,
    "mocap": _plot_mocap,
}


def show_keyframe(clip, view, title=None):
    """Display one modality for the notebook.

    Depth, Seek Thermal, IRA, and ToF are the color image at the middle depth
    frame, with no axis names. WiFi draws one heatmap per transmitter. The
    mmWave cloud and the skeleton are 3D. Heatmaps and line plots use the
    stream's own time stamps on the horizontal axis. The mp4 views are unchanged.
    """
    from IPython.display import display

    known = list(VIEWS)
    if view not in VIEWS:
        raise ValueError(f"Unknown view: {view}. Choose from {', '.join(known)}")
    if view in _IMAGE_VIEWS:
        image = render_keyframes(clip, views=[view])[view]
        height, width = image.shape[:2]
        fig, ax = plt.subplots(figsize=(8, max(2.5, 8 * height / width)))
        ax.imshow(image)
        ax.axis("off")
        if title:
            ax.set_title(title)
        fig.tight_layout()
    else:
        fig = _NOTEBOOK_PLOTS[view](clip, title)
    display(fig)
    plt.close(fig)


def visualize_recording(
    recording=None,
    output_dir=DEFAULT_OUTPUT,
    segment=0,
    max_frames=48,
    fps=10,
    views=None,
    node_id=1,
):
    """Load the prepared recording, if needed, and visualize one or every middle cut.

    ``segment=None`` writes every middle cut. Only ``node_id`` is loaded. A
    modality that exists on exactly one other node is still included. RGB is not read.
    """
    if recording is None:
        recording = load_recording(node_id=node_id)
    if segment is None:
        indexes = range(len(recording["segments"]))
    else:
        indexes = [segment]
    written = []
    for index in indexes:
        start, end = recording["segments"][index]
        clip = _slice_recording(recording, start, end)
        folder = Path(output_dir) / f"{clip['activity']}_user{clip['user_id']}_seg{index:02d}"
        print(f"segment {index}: {start} -> {end}")
        written.extend(
            visualize_clip(clip, folder, max_frames=max_frames, fps=fps, views=views)
        )
    return written


def main():
    parser = argparse.ArgumentParser(description="Write OctoNet demo views for one recording.")
    parser.add_argument("--segment", type=int, default=0, help="Middle-cut index. Default: 0.")
    parser.add_argument("--all-segments", action="store_true", help="Write every middle cut.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=48,
        help="Camera frames to draw. 0 keeps every frame. Default: 48.",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--views",
        nargs="*",
        default=None,
        help=f"Subset of views. Default: all. Choices: {', '.join(VIEWS)}",
    )
    args = parser.parse_args()
    visualize_recording(
        output_dir=args.output,
        segment=None if args.all_segments else args.segment,
        max_frames=args.max_frames,
        fps=args.fps,
        views=args.views,
    )


if __name__ == "__main__":
    main()
