# Code for parsing cheeseboard timestamp CSV files into structured trial data.
# Given an input timestamp file, organize the data using created classes to structure the data.
# In the future, this will also be useful for linking any timestamped data to its specific experiment trial.
import pandas as pd

import matplotlib.pyplot as plt

import os

class Trial:
    def __init__(self, trial_dict):
        # Basic trial information
        self.session_id = trial_dict.get('session_id', None)
        self.trial_number = trial_dict.get('trial_number', None)

        # Start time (UNIX) and index from timestamp file
        self.start_idx = trial_dict.get('start_idx', None)
        self.start_time = trial_dict.get('start_time', None)

        # End time (UNIX) and index from timestamp file
        self.end_idx = trial_dict.get('end_idx', None)
        self.end_time = trial_dict.get('end_time', None)

        # Duration of the trial 
        self.duration = trial_dict.get('duration', None)

        # Reward times and indices
        self.sb_time = trial_dict.get('sb_time', None)
        self.sb_idx = trial_dict.get('sb_idx', None)
        self.R1_time = trial_dict.get('R1_time', None)
        self.R1_idx = trial_dict.get('R1_idx', None)
        self.R2_time = trial_dict.get('R2_time', None)
        self.R2_idx = trial_dict.get('R2_idx', None)
        self.R3_time = trial_dict.get('R3_time', None)
        self.R3_idx = trial_dict.get('R3_idx', None)
        
        # Derived metrics from reward times
        self.first_time = trial_dict.get('first_time', None)
        self.first_idx = trial_dict.get('first_idx', None)
        self.last_time = trial_dict.get('last_time', None)
        self.last_idx = trial_dict.get('last_idx', None)

        # If any of the values are missing, print the trial number.
        for key, value in trial_dict.items():
            if value is None:
                print(f"Warning: Trial {self.trial_number} missing value for {key}")
        
    def __repr__(self):
        return f"Trial {self.trial_number}: Start {self.start_time}, End {self.end_time}, Duration {self.duration}"

class ExperimentStruct:
    def __init__(self, timestamp_file):
        # Basic experiment information
        self.experiment_type = None

        # Timestamp conversion
        self.timestamp_file = timestamp_file
        self.trials = self.convert_to_trials()

    def convert_to_trials(self):
        experiment = pd.read_csv(self.timestamp_file, header=None)
        trials = []

        # From the csv, find all rows where the value is not 0
        event_rows = experiment[experiment[2] != 0]

        # Read in column 0 as datetime in UNIX time
        event_rows.loc[:, 0] = pd.to_datetime(event_rows.loc[:, 0], unit='ns')

        # Ideally, the data should be structured where each trial
        # starts with a 1 (start)
        # indicates leaving the rest box (2)
        # 3 rewards (3, 4, 5)
        # ends with a 6 (end)
        current_trial = None
        trial_number = 0
        for idx, row in event_rows.iterrows():
            unix_time = row[0]
            state = row[2]

            if state == 1:  # Start of trial
                # Start a new trial
                current_trial = {
                    'start_idx': idx,
                    'start_time': unix_time,
                    'trial_number': trial_number,
                    'R1_time': None,
                    'R1_idx': None,
                    'R2_time': None,
                    'R2_idx': None,
                    'R3_time': None,
                    'R3_idx': None
                }
            elif state == 2 and current_trial is not None:  # Left rest box
                current_trial['sb_time'] = unix_time
                current_trial['sb_idx'] = idx

            elif state == 3 and current_trial is not None:  # R1 reached
                current_trial['R1_time'] = unix_time
                current_trial['R1_idx'] = idx

            elif state == 4 and current_trial is not None:  # R2 reached
                current_trial['R2_time'] = unix_time
                current_trial['R2_idx'] = idx

            elif state == 5 and current_trial is not None:  # R3 reached
                current_trial['R3_time'] = unix_time
                current_trial['R3_idx'] = idx

            elif state == 6 and current_trial is not None:  # End of trial
                current_trial['end_time'] = unix_time
                current_trial['end_idx'] = idx

                # Record the last reward index as an end time
                current_trial['first_time'] = min(filter(None, [current_trial['R1_time'], current_trial['R2_time'], current_trial['R3_time']]))
                current_trial['first_idx'] = min(filter(None, [current_trial['R1_idx'], current_trial['R2_idx'], current_trial['R3_idx']]))
                current_trial['last_time'] = max(filter(None, [current_trial['R1_time'], current_trial['R2_time'], current_trial['R3_time']]))
                current_trial['last_idx'] = max(filter(None, [current_trial['R1_idx'], current_trial['R2_idx'], current_trial['R3_idx']]))
                current_trial['duration'] = (current_trial['last_time'] - current_trial['first_time']).total_seconds()
                trial_number += 1
                trial_data = Trial(current_trial)
                trials.append(trial_data)
                current_trial = None

        return trials

    def findSLEAPFile(self, timestamp_file):
        # Assuming the SLEAP file is in the same directory as the timestamp file
        base_dir = os.path.dirname(timestamp_file)
        for file in os.listdir(base_dir):
            if file.endswith('.analysis.csv'):
                return os.path.join(base_dir, file)
        return None
    

timestamp = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/ExperimentVideo_2025-11-10_0705_timestamps.csv'
trial_data = ExperimentStruct(timestamp)

# Plot trial durations
durations = [trial.duration for trial in trial_data.trials]
plt.figure(figsize=(10, 6))
plt.plot(durations, marker='o')
plt.title('Trial Durations')
plt.xlabel('Trial Number')
plt.ylabel('Duration (seconds)')
plt.show()