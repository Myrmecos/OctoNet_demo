# rename node_1_vayyar_data_path column's data paths according to rename_vayyar.json
import json
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# load rename_vayyar.json
with open('rename_vayyar.json', 'r') as f:
    rename_vayyar = json.load(f)

# load node_1_vayyar_data_path column's data paths
df = pd.read_csv('cut_manual_with_new_timestamp.csv')
node_1_vayyar_data_path = df['node_1_vayyar_data_path']

# rename node_1_vayyar_data_path column's data paths according to rename_vayyar.json
for i in range(len(node_1_vayyar_data_path)):
    if pd.isna(node_1_vayyar_data_path[i]):
        print("one na")
        continue
    name = os.path.basename(node_1_vayyar_data_path[i])
    # print("DEBUG: name is:", name)
    if name in rename_vayyar:
        node_1_vayyar_data_path[i] = f"node_1/{rename_vayyar[name]}.pickle"
    else:
        print(f"Data path {node_1_vayyar_data_path[i]} not found in rename_vayyar.json")
    # break

# save the updated dataframe
df['node_1_vayyar_data_path'] = node_1_vayyar_data_path
df.to_csv('cut_manual_with_new_timestamp_vayyar_updated.csv', index=False)