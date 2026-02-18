# Given a path of data, generate a list of all file locations split by parent folder
# The assumed structure is that each experiment is in its own folder,
# and we want the ability to combine per day or per condition

import os

search_folder = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/COMBINED/'

# Per folder, add all _timestamps.csv files to a list
session_dict = {}
for root, dirs, files in os.walk(search_folder):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        timestamp_files = []
        for file in os.listdir(dir_path):
            if file.endswith('_timestamps.csv'):
                timestamp_files.append(os.path.join(dir_path, file))
        if timestamp_files:
            session_dict[dir_name] = timestamp_files

# Save to a YAML file
save_path = os.path.join(search_folder, 'combined_sessions.yaml')
with open(save_path, 'w') as f:
    f.write('sessions:\n')
    for session, files in session_dict.items():
        for file in files:
            f.write(f'  - {file}\n')