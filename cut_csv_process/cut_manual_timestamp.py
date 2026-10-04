# add timestamp to cut_manual.csv, get cut_manual_with_new_timestamp.csv
# 1. read cut_manual.csv, extract node_1_IRA_data_path column
# 2. read timestamps.txt, each line is a timestamp
# 3. for each node_1_IRA_data_path, find the corresponding timestamp
# 4. write the corresponding timestamp to a new column new_timestamp in cut_manual.csv

import pandas as pd

def timediff(timestr1, timestr2):
    # timestr format: 20260710134226: yyyymmddhhmmss
    from datetime import datetime
    dt1 = datetime.strptime(timestr1, "%Y%m%d%H%M%S")
    dt2 = datetime.strptime(timestr2, "%Y%m%d%H%M%S")
    return abs((dt1 - dt2).total_seconds())

if __name__=="__main__":
    df = pd.read_csv("cut_manual.csv")
    timestamps = [line.strip() for line in open("timestamps.txt")]
    # print("Number of timestamps:", len(timestamps))
    print("Number of node_1_IRA_data_path entries:", len(df["node_1_IRA_data_path"]))
    print(f"timestamp example: '{timestamps[0]}'")

    for index, row in df.iterrows():
        ira_timestamp_path = row['node_1_IRA_data_path']
        ira_timestamp = ira_timestamp_path.split("/")[-1].split(".")[0].split("_")[0]
        # print(f"ira_timestamp: '{ira_timestamp}'")
        # break
        for timestamp in timestamps:
            if timediff(timestamp, ira_timestamp) < 25:
                df.at[index, 'new_timestamp'] = timestamp
                break

    #print num of uniq vals in new_timestamp
    print("Number of unique new_timestamp entries:", len(df["new_timestamp"].unique()))

    # save to new csv
    df.to_csv("cut_manual_with_new_timestamp.csv", index=False)

