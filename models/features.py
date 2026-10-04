"""Turn one activity window into a fixed CHW tensor for ResNet-18.

Each modality keeps the array layout produced by ``data_loader_new`` and is
resampled along time (and, where needed, along its other axis) so every sample
of that modality has the same shape. Values are per-sample standardized.
"""

import numpy as np

# (channels, height, width). Time is either the channel axis or the height axis.
SHAPES = {
    "IRA": (16, 24, 32),
    "wifi": (2, 48, 114),
    "uwb": (1, 48, 256),
    "mmWave": (4, 16, 32),
    "ToF": (16, 8, 8),
    "polar": (1, 32, 64),
    "vayyar": (1, 48, 400),
    "seekThermal": (8, 48, 64),
    "depthCamera": (8, 48, 64),
    "acoustic": (1, 128, 64),
    "imu": (1, 48, 221),
    "mocap": (3, 48, 20),
}


def standardize(array):
    values = np.nan_to_num(np.asarray(array, dtype=np.float64))
    centered = values - float(values.mean())
    std = float(centered.std())
    if std < 1e-6:
        return centered.astype(np.float32)
    return (centered / std).astype(np.float32)


def resample_axis(array, length, axis=0):
    """Linearly resample ``array`` so ``axis`` has ``length`` entries."""
    array = np.asarray(array, dtype=np.float32)
    if array.shape[axis] == length:
        return array
    moved = np.moveaxis(array, axis, 0)
    if moved.shape[0] == 0:
        raise ValueError("cannot resample an empty axis")
    index = np.linspace(0, moved.shape[0] - 1, length)
    left = np.clip(np.floor(index).astype(np.int64), 0, moved.shape[0] - 1)
    right = np.minimum(left + 1, moved.shape[0] - 1)
    weight = (index - left).astype(np.float32).reshape((length,) + (1,) * (moved.ndim - 1))
    mixed = (1.0 - weight) * moved[left] + weight * moved[right]
    return np.moveaxis(mixed, 0, axis)


def _take_indices(count, length):
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.clip(np.linspace(0, count - 1, length).round().astype(np.int64), 0, count - 1)


def _resize_hw(frame, height, width):
    import cv2

    frame = np.asarray(frame, dtype=np.float32)
    if frame.ndim == 3:
        frame = frame[:, :, 0]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def from_ira(frames):
    channels, height, width = SHAPES["IRA"]
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or frames.shape[0] == 0:
        raise ValueError(f"IRA frames have shape {frames.shape}")
    if frames.shape[1:] != (height, width):
        frames = np.stack([_resize_hw(frame, height, width) for frame in frames])
    return standardize(resample_axis(frames, channels, axis=0))


def from_wifi(frames):
    channels, time_steps, subcarriers = SHAPES["wifi"]
    amplitude = np.abs(np.asarray(frames)).astype(np.float32)
    if amplitude.ndim != 3 or amplitude.shape[0] == 0:
        raise ValueError(f"wifi frames have shape {amplitude.shape}")
    amplitude = resample_axis(amplitude, time_steps, axis=0)
    amplitude = resample_axis(amplitude, subcarriers, axis=2)
    # (T, transmitters, subcarriers) -> (transmitters, T, subcarriers)
    ordered = np.transpose(amplitude, (1, 0, 2))
    if ordered.shape[0] < channels:
        pad = np.zeros((channels - ordered.shape[0],) + ordered.shape[1:], dtype=np.float32)
        ordered = np.concatenate([ordered, pad], axis=0)
    return standardize(ordered[:channels])


def from_uwb(frames):
    _, time_steps, bins = SHAPES["uwb"]
    cir = np.abs(np.asarray(frames)).astype(np.float32)
    if cir.ndim == 1:
        cir = cir.reshape(1, -1)
    if cir.ndim != 2 or cir.shape[0] == 0:
        raise ValueError(f"uwb frames have shape {getattr(frames, 'shape', None)}")
    cir = resample_axis(cir, time_steps, axis=0)
    cir = resample_axis(cir, bins, axis=1)
    return standardize(cir.reshape(1, time_steps, bins))


def from_mmwave(points):
    channels, time_steps, point_count = SHAPES["mmWave"]
    if len(points) == 0:
        raise ValueError("mmWave window has no frames")
    chosen = _take_indices(len(points), time_steps)
    grid = np.zeros((time_steps, point_count, channels), dtype=np.float32)
    for row, frame_index in enumerate(chosen):
        cloud = np.asarray(points[int(frame_index)], dtype=np.float32)
        if cloud.size == 0:
            continue
        if cloud.ndim == 1:
            cloud = cloud.reshape(1, -1)
        cloud = cloud[:, :channels]
        if cloud.shape[1] < channels:
            pad = np.zeros((cloud.shape[0], channels - cloud.shape[1]), dtype=np.float32)
            cloud = np.concatenate([cloud, pad], axis=1)
        take = _take_indices(cloud.shape[0], min(point_count, cloud.shape[0]))
        grid[row, : len(take)] = cloud[take]
    # (T, points, xyzv) -> (xyzv, T, points)
    return standardize(np.transpose(grid, (2, 0, 1)))


def from_tof(tof_bins):
    channels, height, width = SHAPES["ToF"]
    bins = np.asarray(tof_bins, dtype=np.float32)
    if bins.ndim != 4 or bins.shape[0] == 0:
        raise ValueError(f"ToF bins have shape {bins.shape}")
    # Average the 18 range bins, leaving an 8x8 spatial map per frame.
    spatial = bins.mean(axis=-1)
    spatial = resample_axis(spatial, channels, axis=0)
    if spatial.shape[1:] != (height, width):
        spatial = np.stack([_resize_hw(frame, height, width) for frame in spatial])
    return standardize(spatial)


def from_polar(frames):
    _, height, width = SHAPES["polar"]
    series = np.asarray(frames, dtype=np.float32).reshape(-1)
    if series.size == 0:
        raise ValueError("polar window is empty")
    series = resample_axis(series.reshape(-1, 1), width, axis=0).reshape(-1)
    image = np.repeat(series.reshape(1, width), height, axis=0)
    return standardize(image.reshape(1, height, width))


def from_vayyar(frames):
    _, time_steps, antennas = SHAPES["vayyar"]
    cube = np.abs(np.asarray(frames)).astype(np.float32)
    if cube.ndim != 3 or cube.shape[0] == 0:
        raise ValueError(f"vayyar frames have shape {cube.shape}")
    # Average the ADC axis. Rows of the image are time, columns are antennas.
    profile = cube.mean(axis=-1)
    profile = resample_axis(profile, time_steps, axis=0)
    profile = resample_axis(profile, antennas, axis=1)
    return standardize(profile.reshape(1, time_steps, antennas))


def from_images(frames, modality):
    channels, height, width = SHAPES[modality]
    frames = np.asarray(frames)
    if frames.ndim == 2:
        frames = frames.reshape(1, *frames.shape)
    if frames.shape[0] == 0:
        raise ValueError(f"{modality} window has no frames")
    chosen = frames[_take_indices(frames.shape[0], channels)]
    resized = np.stack([_resize_hw(frame, height, width) for frame in chosen])
    return standardize(resized)


def from_acoustic(mel):
    _, mels, time_steps = SHAPES["acoustic"]
    spectrum = np.asarray(mel, dtype=np.float32)
    if spectrum.ndim != 2 or spectrum.shape[1] == 0:
        raise ValueError(f"acoustic mel has shape {spectrum.shape}")
    spectrum = resample_axis(spectrum, mels, axis=0)
    spectrum = resample_axis(spectrum, time_steps, axis=1)
    return standardize(spectrum.reshape(1, mels, time_steps))


def from_imu(frames):
    _, time_steps, features = SHAPES["imu"]
    values = np.asarray(frames, dtype=np.float32)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError(f"imu frames have shape {values.shape}")
    flat = values.reshape(values.shape[0], -1)
    if flat.shape[1] < features:
        pad = np.zeros((flat.shape[0], features - flat.shape[1]), dtype=np.float32)
        flat = np.concatenate([flat, pad], axis=1)
    flat = resample_axis(flat[:, :features], time_steps, axis=0)
    return standardize(flat.reshape(1, time_steps, features))


def from_mocap(positions):
    channels, time_steps, joints = SHAPES["mocap"]
    pose = np.asarray(positions, dtype=np.float32)
    if pose.ndim != 3 or pose.shape[0] == 0:
        raise ValueError(f"mocap positions have shape {pose.shape}")
    if pose.shape[1] < joints:
        pad = np.zeros((pose.shape[0], joints - pose.shape[1], pose.shape[2]), dtype=np.float32)
        pose = np.concatenate([pose, pad], axis=1)
    pose = pose[:, :joints, :channels]
    if pose.shape[2] < channels:
        pad = np.zeros((pose.shape[0], pose.shape[1], channels - pose.shape[2]), dtype=np.float32)
        pose = np.concatenate([pose, pad], axis=2)
    pose = resample_axis(pose, time_steps, axis=0)
    # (T, joints, xyz) -> (xyz, T, joints)
    return standardize(np.transpose(pose, (2, 0, 1)))
