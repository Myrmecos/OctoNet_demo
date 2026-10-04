# update the data path of cut_manual_with_new_timestamp.csv according to the file list

# information needed for updating the data path of cut_manual_with_new_timestamp.csv
# 1. node id
# 2. modality id
# 3. timestamp
# note: mind the diff between mmWave and mmwave

import os

import pandas as pd

# ================================ build index from file list ================================
# Helper functions
def load_file_list(path = "file_path_list.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f]

# use node_id, modality_id and timestamp to build index
def build_index(file_path):
    filename = os.path.basename(file_path).split(".")[0]
    fields = filename.split("_")
    node_id = fields[0]
    modality_id = fields[1]
    timestamp = fields[6]
    return node_id, modality_id, timestamp

# build a dict indexed by node_id, modality_id and timestamp, returns file path
def build_index_from_list(file_list):
    index_dict = {}
    for file_path in file_list:
        node_id, modality_id, timestamp = build_index(file_path)
        index_dict[(node_id, modality_id, timestamp)] = file_path
    return index_dict

# ================================ modify csv file ================================
# TODO: 
# 1. for every csv row, extract timestamp
# 2. for every modality on different node (data path column), extract node_id, modality_id
# 3. use the index_dict to find the file path
# 4. update the data path column (if found in index_dict, otherwise keep the original value)

# information about the csv:
# activity: contains activity
# new_timestamp: contains timestamp
# uwb_data_path: modality: uwb; node: 1
# node_1_IRA_data_path': modality: ira; node: 1
# node_1_uwb_data_path': modality: uwb; node: 1
# node_1_wifi_data_path': modality: wifi; node: 1
# node_1_mmWave_data_path: modality: mmWave; node: 1
# node_1_polar_data_path: modality: heartrate; node: 1
# node_1_acoustic_data_path: modality: acoustic; node: 1
# node_1_seekThermal_data_path: modality: seekthermal; node: 1
# node_1_depthCamera_data_path: modality: depthcam; node: 1
# node_2_IRA_data_path: modality: ira; node: 2
# node_2_wifi_data_path: modality: wifi; node: 2
# node_2_mmWave_data_path: modality: mmwave; node: 2
# node_3_IRA_data_path: modality: ira; node: 3
# node_3_wifi_data_path: modality: wifi; node: 3
# node_3_mmWave_data_path: modality: mmwave; node: 3
# node_3_seekThermal_data_path: modality: seekthermal; node: 3
# node_3_depthCamera_data_path: modality: depthcam; node: 3
# node_4_IRA_data_path: modality: ira; node: 4
# node_4_wifi_data_path: modality: wifi; node: 4
# node_4_mmWave_data_path: modality: mmwave; node: 4
# node_4_ToF_data_path: modality: ToF; node: 4
# node_4_acoustic_data_path: modality: acoustic; node: 4
# node_5_IRA_data_path: modality: ira; node: 5
# node_5_mmWave_data_path: modality: mmwave; node: 5
# node_5_depthCamera_data_path: modality: depthcam; node: 5
# mocap_data_path: modality: mocapcsv; node: x
#   (file list uses node id "x", e.g. node_x/x_mocapcsv_...)
# imu_data_path: modality: imu; node: 1
#   (file list uses node id "1", e.g. node_1/1_imu_...)

# YOUR TASK: update data columns (paths only) of cut_manual_with_new_timestamp.csv
# column name -> (node_id, modality_id) as stored in the file-list index.
# Only node 1 spells the modality "mmWave"; nodes 2-5 use "mmwave".
PATH_COLUMNS = {
    "uwb_data_path": ("1", "uwb"),
    "node_1_IRA_data_path": ("1", "ira"),
    "node_1_uwb_data_path": ("1", "uwb"),
    "node_1_wifi_data_path": ("1", "wifi"),
    "node_1_mmWave_data_path": ("1", "mmWave"),
    "node_1_polar_data_path": ("1", "heartrate"),
    "node_1_acoustic_data_path": ("1", "acoustic"),
    "node_1_seekThermal_data_path": ("1", "seekthermal"),
    "node_1_depthCamera_data_path": ("1", "depthcam"),
    "node_2_IRA_data_path": ("2", "ira"),
    "node_2_wifi_data_path": ("2", "wifi"),
    "node_2_mmWave_data_path": ("2", "mmwave"),
    "node_3_IRA_data_path": ("3", "ira"),
    "node_3_wifi_data_path": ("3", "wifi"),
    "node_3_mmWave_data_path": ("3", "mmwave"),
    "node_3_seekThermal_data_path": ("3", "seekthermal"),
    "node_3_depthCamera_data_path": ("3", "depthcam"),
    "node_4_IRA_data_path": ("4", "ira"),
    "node_4_wifi_data_path": ("4", "wifi"),
    "node_4_mmWave_data_path": ("4", "mmwave"),
    "node_4_ToF_data_path": ("4", "ToF"),
    "node_4_acoustic_data_path": ("4", "acoustic"),
    "node_5_IRA_data_path": ("5", "ira"),
    "node_5_mmWave_data_path": ("5", "mmwave"),
    "node_5_depthCamera_data_path": ("5", "depthcam"),
    "mocap_data_path": ("x", "mocapcsv"),
    "imu_data_path": ("1", "imu"),
    # "node_1_vayyar_data_path": ("1", "vayyarmmwave"),
}

def timestamp_key(value):
    if pd.isna(value):
        return None
    return str(value).split(".")[0]

def update_data_paths(df, index_dict):
    """Replace path cells when (node_id, modality_id, new_timestamp) is in the index."""
    stats = {"updated": 0, "filled_empty": 0, "kept": 0}
    for col, (node_id, modality_id) in PATH_COLUMNS.items():
        for index, row in df.iterrows():
            timestamp = timestamp_key(row["new_timestamp"])
            if timestamp is None:
                stats["kept"] += 1
                continue
            new_path = index_dict.get((node_id, modality_id, timestamp))
            if new_path is None:
                stats["kept"] += 1
                continue
            original = row[col]
            if pd.isna(original) or str(original).strip() == "":
                stats["filled_empty"] += 1
            stats["updated"] += 1
            df.at[index, col] = new_path
    return stats


if __name__ == "__main__":
    file_list = load_file_list()
    index_dict = build_index_from_list(file_list)
    print(f"Number of index entries: {len(index_dict)}") # 25283

    csv_path = "cut_manual_with_new_timestamp.csv"
    df = pd.read_csv(csv_path, dtype={"new_timestamp": str})
    stats = update_data_paths(df, index_dict)
    df.to_csv(csv_path, index=False)
    print(
        f"Updated path cells: {stats['updated']} "
        f"(filled previously empty: {stats['filled_empty']}); "
        f"kept original: {stats['kept']}"
    )