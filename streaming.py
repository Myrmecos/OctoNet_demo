import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"
# os.environ["TQDM_DISABLE"] = "0"
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
from huggingface_hub import list_repo_files, snapshot_download, hf_hub_download, whoami
import argparse
import json
from pathlib import Path
from tqdm import tqdm
print(whoami())

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.json"
DEFAULT_LOCAL_DIR = ROOT / "downloaded_data"

def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    return config_dict

def parse_filename(filepath):
    basename = Path(filepath).name
    name_without_ext = basename.rsplit(".", 1)[0]
    fields = name_without_ext.split("_")

    if len(fields) != 7:
        print("filename format error:", filepath)
        return None

    node_id, modality_id, scene_id, user_id, activity_id, trial_id, timestamp = fields
    return {
        "node_id": node_id,
        "modality_id": modality_id,
        "scene_id": scene_id,
        "user_id": user_id,
        "activity_id": activity_id,
        "trial_id": trial_id,
        "timestamp": timestamp, 
        "full_path": filepath
    }


def match_filters(parsed, filters):
    if parsed is None:
        return False
    """检查是否满足所有过滤条件"""
    for key, allowed in filters.items():
        if allowed is None or allowed == []:      # 不限制该字段
            continue
        if parsed.get(key) not in allowed:
            # print("end=", parsed.get(key), "not in allowed", allowed)
            return False
    return True
    
def parse_args():
    parser = argparse.ArgumentParser(
        description="Download OctoNet files that match a JSON filter config."
    )
    parser.add_argument(
        "--config_path",
        default=str(DEFAULT_CONFIG),
        help=f"JSON filter config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--local_dir",
        default=str(DEFAULT_LOCAL_DIR),
        help=f"Directory for downloaded files. Default: {DEFAULT_LOCAL_DIR}",
    )
    return parser.parse_args()


# ========= main logic =========
if __name__=="__main__":
    args = parse_args()
    config_path = args.config_path
    local_dir = args.local_dir
    filters = load_config(config_path)

    # list all files in repo
    repo_id = "hku-aiot/OctoNet"
    files = list_repo_files(repo_id, repo_type="dataset")

    # prepare target files
    target = []
    for f in tqdm(files):
        parsed = parse_filename(f)
        if parsed is None:
            continue
        if match_filters(parsed, filters):
            target.append(f)
    print(f"found {len(target)} matching files")

    # write the files to download
    with open("octonet_download_target.txt", "w") as f:
        for item in target:
            f.write("%s\n" % item)

    # config file to specify which files to download
    # json as config
    patterns = [f"*{Path(f).name}*" for f in target]

    # download
    for i, filename in enumerate(target, 1):
        print(f"[{i}/{len(target)}] {filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )