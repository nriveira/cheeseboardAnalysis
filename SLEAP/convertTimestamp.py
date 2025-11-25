# Input
# - timestamp file from Cherubim to split into trials
# Structure of CSV:
# - No header
# - Columns:
#   - Column 0: UNIX timestamp
#   - Column 1: Computer time stamp
#   - Column 2: State of experiment
#     - 0: No change
#     - 1: Start of trial
#     - 2: Left rest box
#     - 3: R1 reached
#     - 4: R2 reached
#     - 5: R3 reached
#     - 6: End of trial

# Output
# Structured list of per trial information:
# - list of dicts with keys:
#   - 'start_idx': int
#   - 'end_idx': int
#   - 'start_time': float
#   - 'end_time': float
#   - 'duration': float
#   - 'trial_number': int
#   - 'R1_time': float
#   - 'R2_time': float
#   - 'R3_time': float

import pandas as pd
import datetime
import os

class Timestamp():
    def convert_timestamp(timestamp_file):
        # Read the timestamp CSV file
        df = pd.read_csv(timestamp_file, header=None)

        # Parse through the DataFrame to extract trial information
        trials = []
        current_trial = None
        trial_number = 0
        for idx, row in df.iterrows():
            unix_time = datetime.datetime.fromtimestamp(int(row[1] / 10e9)) # Convert nanoseconds to seconds
            state = row[2]

            if state == 1:  # Start of trial
                if current_trial is not None:
                    # If there's an ongoing trial, finalize it
                    current_trial['end_idx'] = idx
                    current_trial['end_time'] = unix_time
                    current_trial['duration'] = current_trial['end_time'] - current_trial['start_time']
                    trials.append(current_trial)

                # Start a new trial
                trial_number += 1
                current_trial = {
                    'start_idx': idx,
                    'start_time': unix_time,
                    'trial_number': trial_number,
                    'R1_time': None,
                    'R2_time': None,
                    'R3_time': None
                }

            elif state == 3 and current_trial is not None:  # R1 reached
                current_trial['R1_time'] = unix_time

            elif state == 4 and current_trial is not None:  # R2 reached
                current_trial['R2_time'] = unix_time

            elif state == 5 and current_trial is not None:  # R3 reached
                current_trial['R3_time'] = unix_time

            elif state == 6 and current_trial is not None:  # End of trial
                current_trial['end_idx'] = idx
                current_trial['end_time'] = unix_time
                current_trial['duration'] = current_trial['end_time'] - current_trial['start_time']
                trials.append(current_trial)
                current_trial = None

        return trials
    
    def findSLEAPFile(timestamp_file):
        # Derive SLEAP file path from timestamp file path, given they are in the same place and have
        # similar names: 
        # Timestamp: ExperimentVideo_YYYY-MM-DD_hhmm_timestamps.csv
        # SLEAP: labelsNov17.v003.XXX_ExperimentVideo_YYYY-MM-DD_hhmm.analysis.csv
        dir_path = os.path.dirname(timestamp_file)
        # Find any files that start with labelsNov17.v003. and end with .analysis.csv
        # and contain the same ExperimentVideo_YYYY-MM-DD_hhmm part
        for file in os.listdir(dir_path):
            if file.startswith("labelsNov17.v003.") and file.endswith(".analysis.csv"):
                if "ExperimentVideo" in file:
                    if timestamp_file.split("ExperimentVideo")[1].split("_timestamps.csv")[0] in file:
                        sleap_file_path = os.path.join(dir_path, file)
                        return sleap_file_path

        raise FileNotFoundError(f"SLEAP file not found for: {timestamp_file}")