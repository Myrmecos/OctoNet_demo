# introduction to current repo
This is an introduction to current repository.

**For a quick start, go to `demo.ipynb`.**

# Intro to files and directories
## `demo.ipynb`
A quick start to the OctoNet's data downloading, loading and visualization.

## `cut_csv_process/`
1. contains code and intermediate results for updating cut_manual.csv from previous version
2. not used in visualization/data loading

## `mocap_processing/`
1. Processes raw mocap data and output poses (key points).

## `models/`; `models.py`
1. baseline models for OctoNet.

<!-- ## `downloaded_octonet_oneuser/`
1. downloaded data for visualization -->
## `OctoNet/`
1. OctoNet official GitHub repo, contains old data loading and visualization code.

<!-- ## `unused`
1. unused files. irrelevant but should be kept for now. -->
<!-- ## `viz_output`
1. visualization output, if any, can be saved here. -->
## `config_oneuser.json`
1. Example config file for data download. 

## `cut_manual.csv`
1. Metadata about the files. Note that this cut_manual.csv is updated to adapt to the updated filenames and file paths of OctoNet.

## `mocap_convert.py`
1. Converts mocap.csv into pkl pose.

## `streaming.py`
1. For downloading the data according to the config file.

## `unzip_commands.sh`
1. For unzipping downloaded zip files.

## `data_loader_new.py`
Loads the prepared recording in `downloaded_octonet_oneuser/` (user 1, `airdrum`, timestamp `20240519182852`) using the matching row in `cut_manual.csv`.

```python
from data_loader_new import load_recording, iter_segments, thermal_to_celsius

# One sensor node. A modality that exists on only one other node is still loaded.
# RGB video is not opened.
recording = load_recording(node_id=1)

# One node of a full-take array, for example IRA (T, 24, 32) or WiFi (T, 2, 114).
ira = recording["IRA"][1]["frames"]

# Depth stays as paths until a cut is requested.
camera = recording["depthCamera"][1]
# camera["depth_paths"], camera["timestamps"]

# Middle activity windows: the first two cuts and the last cut are dropped,
# matching OctoNet/dataset_loader.py.
for start, end, clip in iter_segments(recording):
    depth_m = clip["depthCamera"][1]["depth"]   # float32 meters, shape (T, 480, 640)
    pose = clip["mocap"]["positions"]           # (T, 20, 3), millimeters
    thermal_c = thermal_to_celsius(clip["seekThermal"][1]["frames"])
```

Returned keys:

- Per node: `IRA`, `wifi`, `uwb`, `mmWave`, `ToF`, `polar`, `vayyar`, `seekThermal`, `acoustic`, `depthCamera`
- Single streams: `imu` `(T, 13, 17)`, `mocap` positions `(T, 20, 3)` from `mocap_convert.py`
- `segments`: list of `(start, end)` datetimes
- `missing`: paths named in the CSV that are not on disk

`mmWave` frames are a list of `(N, 4)` arrays (`x, y, z, velocity`). Vayyar frames are complex `(T, 400, 100)`. Acoustic keeps the original wav sample rate and the log start time. Seek Thermal frames are the PNG grayscale values; `thermal_to_celsius` maps them to 15–50 °C. Depth PNGs are mapped to 0–10 m only when a segment is decoded. RGB video is not read. Pass `node_id=1` to load one angle; a modality recorded on exactly one other node, such as ToF on node 4, is still included.

## `data_visualizer_new.py`

Writes one mp4 per demo view for a middle cut of the recording loaded by `data_loader_new.py`. Files go to `viz_output/<activity>_user<id>_seg<index>/`.

```bash
python data_visualizer_new.py
python data_visualizer_new.py --segment 0 --max-frames 48 --fps 10
python data_visualizer_new.py --views depth mocap imu
python data_visualizer_new.py --all-segments
```

```python
from data_loader_new import load_recording, iter_segments
from data_visualizer_new import visualize_clip, visualize_recording

# Segment 0 of the prepared airdrum recording. --max-frames 0 keeps every camera frame.
visualize_recording(segment=0, max_frames=48)

recording = load_recording()
for start, end, clip in iter_segments(recording):
    visualize_clip(clip, "viz_output/one_cut", max_frames=48)
    break
```

The timeline is node-1 depth. Another stream of length `T` is read at `i / (T_depth - 1) * (T - 1)`. `--max-frames` (default 48) subsamples that timeline. RGB video is not read. The recording is loaded for node 1; a modality that exists on exactly one other node is still drawn.

Views, named by the mp4 file:

- `depth`: node 1 depth in meters, scaled per frame.
- `seek`: Seek Thermal node 1, mapped to 15–50 °C.
- `ira`: node 1 temperature map. Unused checkerboard pixels are filled from their neighbors for display.
- `wifi`: node 1 amplitude. The two transmitters are concatenated, and the 40 MHz null subcarriers are removed.
- `tof`: the only ToF node, mean over the 18 bins.
- `mmwave`: node 1 point cloud. Three frames are merged. The panel is x versus z, scaled for the cut; color is forward range y. Empty slots and non-finite returns are left out.
- `vayyar`: node 1 amplitude, mean over the 100 ADC steps. Rows are the 400 antennas.
- `uwb`: node 1 CIR magnitude.
- `acoustic`: node 1 log-mel, after resampling to 16 kHz (`n_fft=1024`, hop 512, 128 mels, `log1p`). Low frequency is at the bottom.
- `imu`: acceleration, magnetic field, Euler angles, and quaternion. Each row draws all 17 markers.
- `polar`: heart rate in BPM.
- `mocap`: 20 joints and the original bone list, from one viewing angle. The plot rotates +90° about Z and scales Z by 10; positions from the loader stay in millimeters.

A white cursor marks the sample aligned with the current depth frame. Heatmaps use one scale for the whole cut. Panels in the mp4 are 320×240; depth in the clip stays 640×480.