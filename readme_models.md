# Per-modality ResNet-18

`models.py` trains one ResNet-18 for activity recognition on each OctoNet sensor stream, then evaluates it on held-out recordings.

Sensor files are read from the `octonet_entire` link (`/disk/banana/Octonet-release/octonet_new`). Nothing in that tree is written or deleted. Metadata comes from `cut_manual.csv`. Windows are the middle activity cuts from `data_loader_new.segment_bounds` (the first two cuts and the last cut are dropped, matching `data_loader_new.py`).

`data_loader_new.py`, `data_visualizer_new.py`, and `demo.ipynb` are unchanged.

## Layout

| Path | Role |
| --- | --- |
| `models.py` | Train and test entry point |
| `models/resnet.py` | ResNet-18 with a configurable input channel count |
| `models/features.py` | Fixed `C×H×W` tensor for each modality |
| `models/data.py` | Recording index, zip decoding, feature cache |
| `models/weights/<modality>_resnet18.pt` | Checkpoint (weights, class list, split, metrics) |
| `models/cache/v2/` | Cached window tensors, so a later epoch or `--test-only` does not reread the raw files |
| `models/results.json` | Accuracy report from the last run |

## Modalities

Each model sees one stream. Node 1 is used when that node recorded the modality. ToF exists only on node 4. IMU and motion capture are single streams. Vayyar pickles are stored under `vayyar_pickle/` even though the CSV lists them under `node_1/`. Depth, Seek Thermal, and acoustic archives are decoded in memory; they are not extracted into `octonet_entire`.

Time is resampled so every window of a modality has the same shape. Each window is standardized on its own.

| Modality | Tensor `C, H, W` | What is encoded |
| --- | --- | --- |
| IRA | 16, 24, 32 | 16 temperature frames as channels |
| wifi | 2, 48, 114 | CSI amplitude, 2 transmitters × 48 times × 114 subcarriers |
| uwb | 1, 48, 256 | CIR magnitude, time × downsampled range bins |
| mmWave | 4, 16, 32 | x, y, z, velocity for 32 points over 16 frames |
| ToF | 16, 8, 8 | 16 frames, mean over the 18 range bins |
| polar | 1, 32, 64 | heart-rate series drawn as a 32×64 strip |
| vayyar | 1, 48, 400 | amplitude, mean over ADC steps, time × antennas |
| seekThermal | 8, 48, 64 | 8 frames mapped to 15–50 °C and resized |
| depthCamera | 8, 48, 64 | 8 frames mapped to 0–10 m and resized |
| acoustic | 1, 128, 64 | log-mel (16 kHz, n_fft 1024, hop 512, 128 mels) |
| imu | 1, 48, 221 | 13 signals × 17 markers, resampled to 48 steps |
| mocap | 3, 48, 20 | x, y, z over time for 20 joints (millimeters) |

## Split and training

Recordings are split by activity, and every window from one recording stays in the same split. The target mix is 70% train, 10% validation, 20% test. An activity with only two recordings keeps one for training and one for test, so there is no validation set and early stopping stays off. An activity with one recording is used only for training.

The loss is class-weighted cross-entropy. The optimizer is AdamW (lr `1e-3`, weight decay `1e-4`) with a cosine learning-rate schedule. The checkpoint with the best validation accuracy is the one that is tested. With no validation set, the checkpoint with the lowest training loss is kept.

## Commands

Use the `octonet` environment. This machine’s GPU driver is older than the installed PyTorch build, so training runs on CPU unless a newer driver is available.

```bash
# Confirm one recording loads and one optimizer step runs for every modality.
python models.py --check

# Train and test every modality on the full metadata table (10 epochs).
python models.py

# Smaller run: two users, three epochs.
python models.py --users 1,2 --epochs 3 --batch-size 16

# One modality, then a later test of the saved weights.
python models.py --modalities IRA,mocap --epochs 20
python models.py --test-only --modalities IRA
```

`--max-recordings N` draws a seeded subset before the split. `--seed` defaults to 41.

## Run recorded in `models/results.json`

All 976 metadata rows, 8 epochs, patience 3, batch size 32, seed 41, CPU. Recordings are stratified by activity into about 695 train / 73 validation / 208 test. There are 64 activity names in `cut_manual.csv`, so a uniform guess is about 1.6%. The checkpoint with the best validation accuracy is the one that was tested.

| Modality | Test accuracy | Macro recall | Test windows |
| --- | ---: | ---: | ---: |
| mocap | 0.671 | 0.657 | 1520 |
| imu | 0.452 | 0.419 | 1514 |
| seekThermal | 0.247 | 0.207 | 1520 |
| depthCamera | 0.210 | 0.170 | 1505 |
| uwb | 0.147 | 0.096 | 1520 |
| ToF | 0.130 | 0.108 | 1520 |
| acoustic | 0.092 | 0.094 | 1520 |
| polar | 0.077 | 0.038 | 1520 |
| vayyar | 0.069 | 0.022 | 1506 |
| wifi | 0.061 | 0.039 | 1497 |
| IRA | 0.048 | 0.032 | 1520 |
| mmWave | 0.017 | 0.016 | 1453 |

A few recordings were left out when the file itself was unreadable: truncated or empty pickles, a depth PNG whose zip entry was empty, and similar. Motion-capture CSVs whose capture start uses `上午` / `下午` are translated to `AM` / `PM` before `mocap_convert` parses them, so those takes are included. `mocap_convert.py` itself is unchanged.
