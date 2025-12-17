# Code for parsing cheeseboard timestamp CSV files into structured trial data.
# Also must consider that SLEAP only reports frames that it successfully tracked,
# so there may be missing frames in the SLEAP data that need to be accounted for
# when linking timestamp data to SLEAP tracking data.

# Given an input timestamp file, organize the data using created classes to structure the data.
# In the future, this will also be useful for linking any timestamped data to its specific experiment trial.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import re

class Trial:
    """ Structure for an individual trial within an experiment """
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

        self.first_time = trial_dict.get('first_time', None)
        self.first_idx = trial_dict.get('first_idx', None)
        self.last_time = trial_dict.get('last_time', None)
        self.last_idx = trial_dict.get('last_idx', None)

        # If any of the values are missing, print the trial number.
        for key, value in trial_dict.items():
            if value is None and key not in ['R1_loc', 'R2_loc', 'R3_loc']:
                print(f"Warning: Trial {self.trial_number} missing value for {key}")

        # Derived values to be filled in with SLEAP data
        self.R1_loc = trial_dict.get('R1_loc', None)
        self.R2_loc = trial_dict.get('R2_loc', None)
        self.R3_loc = trial_dict.get('R3_loc', None)

        self.velocies = None  # To be filled later if needed
        
    def __repr__(self):
        return f"Trial {self.trial_number}: Start {self.start_time}, End {self.end_time}, Duration {self.duration}"

class ExperimentStruct:
    """ Organization of timestampe data into experimental block """
    def __init__(self, timestamp_file):
        # 0) Basic experiment information
        self.experiment_type = None
        # Recognize experiment date from metadata in file name: 
        self.experimentTag = self.extract_experiment_id(timestamp_file)
        print(f"Experiment Tag: {self.experimentTag}")

        # 1) Gather all data associated with experiment block
        self.timestamp_file = timestamp_file
        self.sleap_file = self.findSLEAPFile(timestamp_file)

        # 2) Data preprocessing
        self.trials = self.convert_to_trials()
        self.sleap_data = self.sleap_data_preprocessing()
        self.get_reward_locations()
        # Estimates the distance between nose and neck for scale reference (Assumed 2 cm)
        self.ref_dist = self.estimate_distance()
        
        # 3) Analysis data
        self.durations = self.find_durations() if self.trials else []
        self.pathways = self.find_pathways() if self.sleap_data is not None else []
        self.velocities = self.find_velocity() if self.sleap_data is not None else []
        self.distance_traveled = self.find_distance_traveled() if self.trials and self.velocities is not None else []
        self.head_directions = self.find_head_direction() if self.sleap_data is not None else []
        self.head_angular_velocities = self.find_head_direction_velocity() if self.sleap_data is not None else []

    def extract_experiment_id(self, timestamp_file):
        """Extract the experiment identifier pattern from filename.
        
        Args:
            timestamp_file (str): Path to timestamp file with format ExperimentVideo_YYYY-MM-DD_HHMM_timestamps.csv
            
        Returns:
            str: Experiment identifier (e.g., 'ExperimentVideo_2025-11-10_0705') or None if pattern not found
        """
        filename = os.path.basename(timestamp_file)
        pattern = r'(ExperimentVideo_\d{4}-\d{2}-\d{2}_\d{4})_timestamps\.csv'
        match = re.search(pattern, filename)
        if match:
            return match.group(1)
        else:
            print(f"Warning: Could not extract experiment ID from filename: {filename}")
            return None
            
    def find_matching_files(self, search_directory=None, file_pattern=None):
        """Find all files that contain the experiment identifier.
        
        Args:
            search_directory (str, optional): Directory to search in. Defaults to parent directory of timestamp file.
            file_pattern (str, optional): Additional pattern to match. Defaults to any file containing experiment ID.
            
        Returns:
            list: List of file paths that contain the experiment identifier
        """
        if not self.experimentTag:
            print("No experiment tag available for searching")
            return []
            
        if search_directory is None:
            search_directory = os.path.dirname(self.timestamp_file)
            
        matching_files = []
        for root, dirs, files in os.walk(search_directory):
            for file in files:
                if self.experimentTag in file:
                    if file_pattern is None or re.search(file_pattern, file):
                        matching_files.append(os.path.join(root, file))
                        
        return matching_files

    def sleap_data_preprocessing(self):
        """ Preprocess SLEAP data to match missing frames """
        sleap_data = pd.read_csv(self.sleap_file)
        # Make an empty DataFrame with all of the frame indices from the timestamp file
        timestamp_data = self.experiment_times.index.to_series().reset_index(drop=True)
        # Make an empty DataFrame with all frame indices
        all_frames = pd.DataFrame({'frame_idx': timestamp_data})
        # Merge SLEAP data with all frames to fill in missing frames
        merged_data = pd.merge(all_frames, sleap_data, on='frame_idx', how='left')
        return merged_data
            
    def convert_to_trials(self):
        self.experiment_times = pd.read_csv(self.timestamp_file, header=None)
        trials = []

        # From the csv, find all rows where the value is not 0
        event_rows = self.experiment_times[self.experiment_times[2] != 0]
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
                    'R3_idx': None,
                    'R1_loc': None,
                    'R2_loc': None,
                    'R3_loc': None
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
                current_trial['duration'] = (current_trial['last_time'] - current_trial['sb_time']).total_seconds()
                trial_number += 1
                trial_data = Trial(current_trial)
                trials.append(trial_data)
                current_trial = None

        return trials

    # HELPER FUNCTIONS
    def findSLEAPFile(self, timestamp_file):
        """Find SLEAP analysis file using experiment ID extraction for robust matching"""
        # Extract experiment ID from timestamp filename
        experiment_id = self.extract_experiment_id(timestamp_file)
        if not experiment_id:
            # Fallback to old method if ID extraction fails
            base_dir = os.path.dirname(timestamp_file)
            for file in os.listdir(base_dir):
                if file.endswith('.analysis.csv'):
                    return os.path.join(base_dir, file)
            return None
        
        # Search for SLEAP file containing the experiment ID
        base_dir = os.path.dirname(timestamp_file)
        for file in os.listdir(base_dir):
            if experiment_id in file and file.endswith('.analysis.csv'):
                return os.path.join(base_dir, file)
        
        # If no matching file found, try the old method as fallback
        for file in os.listdir(base_dir):
            if file.endswith('.analysis.csv'):
                print(f"Warning: Using fallback SLEAP file {file} (no experiment ID match found)")
                return os.path.join(base_dir, file)
        
        return None

    # EXTRACTING FUNCTIONS
    def get_reward_locations(self, tracking_part='nose1'):
        """Extract reward locations from trials, assuming the rats location at time of reward is stored"""
        for trial in self.trials:
            print(f"Getting reward locations for Trial {trial.trial_number}")

            # Find the first data point at or after R1_idx
            R1_frame_idx = self.sleap_data[self.sleap_data['frame_idx'] == trial.R1_idx]
            if not R1_frame_idx.empty:
                x = R1_frame_idx[f'{tracking_part}.x'].values
                y = R1_frame_idx[f'{tracking_part}.y'].values
                trial.R1_loc = (x, y)

            R2_frame_idx = self.sleap_data[self.sleap_data['frame_idx'] == trial.R2_idx]
            if not R2_frame_idx.empty:
                x = R2_frame_idx[f'{tracking_part}.x'].values
                y = R2_frame_idx[f'{tracking_part}.y'].values
                trial.R2_loc = (x, y)

            R3_frame_idx = self.sleap_data[self.sleap_data['frame_idx'] == trial.R3_idx]
            if not R3_frame_idx.empty:
                x = R3_frame_idx[f'{tracking_part}.x'].values
                y = R3_frame_idx[f'{tracking_part}.y'].values
                trial.R3_loc = (x, y)

    def find_durations(self):
        """Calculate trial durations in seconds"""
        return [trial.duration for trial in self.trials if trial.duration is not None]
    
    def find_pathways(self, tracking_part='nose1'):
        trials = self.trials
        pathways = []

        for trial in trials:
            print(f"Finding pathway for Trial {trial.trial_number}")
            start_idx = trial.sb_idx if trial.sb_idx is not None else trial.start_idx
            end_idx = trial.last_idx if trial.last_idx is not None else trial.end_idx

            trial_sleap_data = self.sleap_data[(self.sleap_data['frame_idx'] >= start_idx) & (self.sleap_data['frame_idx'] <= end_idx)]
            bodypart_x = f'{tracking_part}.x'
            bodypart_y = f'{tracking_part}.y'

            tracking_x = trial_sleap_data[bodypart_x].values
            tracking_y = trial_sleap_data[bodypart_y].values

            pathways.append((tracking_x, tracking_y))
        return pathways

    def find_velocity(self, tracking_part='nose1', dt=30):
        """Calculate velocities at all points for specified body part"""
        if self.sleap_data is None:
            return []
            
        # Calculate velocities at all points
        bodypart_x = f'{tracking_part}.x'
        bodypart_y = f'{tracking_part}.y'

        tracking_x = self.sleap_data[bodypart_x].values
        tracking_y = self.sleap_data[bodypart_y].values

        # Calculate velocities
        dx = pd.Series(tracking_x).diff().fillna(0).values
        dy = pd.Series(tracking_y).diff().fillna(0).values

        speed = np.sqrt(((dx ** 2 + dy ** 2))) * dt
        adjusted_speed = speed * (2 / self.ref_dist)  # Convert to cm/frame using estimated reference distance
        return adjusted_speed
    
    def find_distance_traveled(self, threshold=75, dt=30):
        """Use velocity data to calculate distance traveled per trial"""
        distances = []
        for trial in self.trials:
            start_idx = trial.sb_idx if trial.sb_idx is not None else trial.start_idx
            end_idx = trial.last_idx if trial.last_idx is not None else trial.end_idx

            trial_velocities = self.velocities[start_idx:end_idx + 1]
            # Remove any velocities above threshold
            trial_velocities = np.where(trial_velocities > threshold, np.nan, trial_velocities)
            distance = np.nansum(trial_velocities)  # Sum of velocities gives distance
            distance = distance / dt # Average distance per frame
            distances.append(distance)
        return distances
    
    def find_head_direction(self):
        """ Using nose and neck positions, calculate head direction angles,
         accounting for wrap-around at 360 degrees """
        if self.sleap_data is None:
            return []

        nose_x = self.sleap_data['nose1.x'].values
        nose_y = self.sleap_data['nose1.y'].values
        neck_x = self.sleap_data['neck1.x'].values
        neck_y = self.sleap_data['neck1.y'].values

        delta_x = nose_x - neck_x
        delta_y = nose_y - neck_y

        head_directions = np.degrees(np.arctan2(delta_y, delta_x)) % 360
        return head_directions
    
    def find_head_direction_velocity(self):
        """ Calculate angular velocity of head direction, specifically 
         accounting for wrap-around at 360 degrees """
        head_directions = self.find_head_direction()

        angular_velocity = np.diff(head_directions, prepend=head_directions[0])
        # Adjust for wrap-around
        angular_velocity = (angular_velocity + 180) % 360 - 180
        return angular_velocity
        
    # SANITY CHECK FUNCTIONS
    def estimate_distance(self, point1='nose1', point2='neck1'):
        """Estimate average distance between two body parts across all frames"""
        if self.sleap_data is None:
            return []

        x1 = self.sleap_data[f'{point1}.x'].values
        y1 = self.sleap_data[f'{point1}.y'].values
        x2 = self.sleap_data[f'{point2}.x'].values
        y2 = self.sleap_data[f'{point2}.y'].values

        distances = np.sqrt(((x1 - x2) ** 2 + (y1 - y2) ** 2))
        avg_distance = np.nanmean(distances)
        # plt.hist(distances, bins=10)
        # plt.axvline(avg_distance, color='red', linestyle='dashed', linewidth=1)
        return avg_distance
    
    def plot_velocity_histogram(self, threshold=75):
        """Plot histogram of velocities with threshold line"""
        if self.velocities is None:
            print("No velocity data available to plot")
            return

        plt.hist(self.velocities, bins=50, color='blue', alpha=0.7)
        plt.axvline(threshold, color='red', linestyle='dashed', linewidth=1)
        plt.title('Velocity Distribution')
        plt.xlabel('Velocity (pixels/frame)')
        plt.ylabel('Frequency')
        plt.show(block=False)

    # PLOTTING FUNCTIONS
    def plot_durations(self):
        """Plot trial durations with moving average"""
        durations = self.durations
        if not durations:
            print("No duration data available to plot")
            return

        # Also do a moving average from the beginning
        window_size = 3
        moving_avg = pd.Series(durations).rolling(window=window_size).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(durations, label='Trial Durations', marker='o')
        plt.plot(moving_avg, color='red', label=f'{window_size}-trial Moving Average')
        plt.legend()
        plt.title('Trial Durations Over Trials')
        plt.xlabel('Trial Number')
        plt.ylabel('Duration (seconds)')
        plt.grid()
        plt.show(block=False)

    def plot_pathways(self, trial=None, tracking_part='neck1'):
        """Plot movement pathways for each trial"""
        if trial is None:
            trials_to_plot = self.trials
        else:
            trials_to_plot = self.trials[trial]

        # Calculate how many rows of 5 plots are needed
        num_cols = (len(trials_to_plot) + 4) // 5

        plt.subplots(num_cols, 5, figsize=(15, 10))
        for num_trial, trial in enumerate(trials_to_plot):
            start_idx = getattr(trial, 'sb_idx', trial.start_idx) if hasattr(trial, 'sb_idx') and trial.sb_idx else trial.start_idx
            end_idx = trial.last_idx if hasattr(trial, 'last_idx') and trial.last_idx else trial.end_idx

            trial_sleap_data = self.sleap_data[(self.sleap_data['frame_idx'] >= start_idx) & (self.sleap_data['frame_idx'] <= end_idx)]
            bodypart_x = f'{tracking_part}.x'
            bodypart_y = f'{tracking_part}.y'

            tracking_x = trial_sleap_data[bodypart_x].values
            tracking_y = trial_sleap_data[bodypart_y].values

            print(f"Indices for Trial {trial.trial_number}: Start {start_idx}, End {end_idx}")

            plt.subplot(num_cols, 5, num_trial + 1)
            plt.plot(tracking_x, tracking_y, label=f'Trial {trial.trial_number}')

            # Also add the reward locations if available
            if trial.R1_loc is not None:
                plt.scatter(trial.R1_loc[0], trial.R1_loc[1], color='red', marker='x', label='R1')
            if trial.R2_loc is not None:
                plt.scatter(trial.R2_loc[0], trial.R2_loc[1], color='green', marker='x', label='R2')
            if trial.R3_loc is not None:
                plt.scatter(trial.R3_loc[0], trial.R3_loc[1], color='blue', marker='x', label='R3')

        plt.title('Movement Pathways Across Trials')
        plt.show(block=False)

    def plot_velocity(self, threshold=75):
        """Plot velocity for each trial"""
        plt.figure(figsize=(15, 10))
        for trial in self.trials:
            start_idx = getattr(trial, 'sb_idx', trial.start_idx) if hasattr(trial, 'sb_idx') and trial.sb_idx else trial.start_idx
            end_idx = trial.last_idx if hasattr(trial, 'last_idx') and trial.last_idx else trial.end_idx

            trial_velocities = self.velocities[start_idx:end_idx + 1]
            # Remove any velocities above threshold
            trial_velocities = np.where(trial_velocities > threshold, np.nan, trial_velocities)

            plt.subplot(4, 5, trial.trial_number + 1)
            plt.plot(trial_velocities, label=f'Trial {trial.trial_number} Velocity')
            plt.title(f'Trial {trial.trial_number} Velocity Over Time')
            plt.xlabel('Frame Index')
            plt.ylabel('Velocity (pixels/frame)')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.show(block=False)

    def plot_distance_traveled(self, threshold=75, dt=30):
        """Plot distance traveled per trial"""
        distances = self.distance_traveled
        if not distances:
            print("No distance data available to plot")
            return

        plt.figure(figsize=(10, 5))
        plt.plot(distances, label='Distance Traveled per Trial', marker='o')
        # Include rolling average per 3 trials
        window_size = 3
        moving_avg = pd.Series(distances).rolling(window=window_size).mean()
        plt.plot(moving_avg, color='red', label=f'{window_size}-trial Moving Average')
        plt.title('Distance Traveled Per Trial')
        plt.xlabel('Trial Number')
        plt.ylabel('Distance Traveled (cm)')
        plt.legend()
        plt.grid()
        plt.show(block=False)

    def plot_head_direction(self):
        """Plot head direction angles over time"""
        head_directions = self.head_directions
        plt.figure(figsize=(15, 10))
        for trial in self.trials:
            start_idx = getattr(trial, 'sb_idx', trial.start_idx) if hasattr(trial, 'sb_idx') and trial.sb_idx else trial.start_idx
            end_idx = trial.last_idx if hasattr(trial, 'last_idx') and trial.last_idx else trial.end_idx

            trial_head_directions = head_directions[start_idx:end_idx + 1]

            plt.subplot(4, 5, trial.trial_number + 1)
            plt.plot(trial_head_directions, label=f'Trial {trial.trial_number} Head Direction')
            plt.title(f'Trial {trial.trial_number} Head Direction Over Time')
            plt.xlabel('Frame Index')
            plt.ylabel('Head Direction (degrees)')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.show(block=False)

    def plot_head_direction_velocity(self):
        """Plot head direction angular velocity over time"""
        angular_velocities = self.head_angular_velocities
        plt.figure(figsize=(15, 10))
        for trial in self.trials:
            start_idx = getattr(trial, 'sb_idx', trial.start_idx) if hasattr(trial, 'sb_idx') and trial.sb_idx else trial.start_idx
            end_idx = trial.last_idx if hasattr(trial, 'last_idx') and trial.last_idx else trial.end_idx

            trial_angular_velocity = angular_velocities[start_idx:end_idx + 1]

            plt.subplot(4, 5, trial.trial_number + 1)
            plt.plot(trial_angular_velocity, label=f'Trial {trial.trial_number} Head Direction Velocity')
            plt.title(f'Trial {trial.trial_number} Head Direction Velocity Over Time')
            plt.xlabel('Frame Index')
            plt.ylabel('Angular Velocity (degrees/frame)')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.show(block=False)

    def plot_angular_velocity_var(self):
        """Plot variance of head angular velocity across trials with box plots"""
        angular_velocities = self.head_angular_velocities
        variances = []
        for trial in self.trials:
            start_idx = getattr(trial, 'sb_idx', trial.start_idx) if hasattr(trial, 'sb_idx') and trial.sb_idx else trial.start_idx
            end_idx = trial.last_idx if hasattr(trial, 'last_idx') and trial.last_idx else trial.end_idx

            trial_angular_velocity = angular_velocities[start_idx:end_idx + 1]
            variance = np.nanvar(trial_angular_velocity)
            variances.append(variance)

        plt.figure(figsize=(10, 5))
        plt.plot(variances, label='Head Angular Velocity Variance per Trial', marker='o')
        plt.title('Head Angular Velocity Variance Across Trials')
        plt.xlabel('Trial Number')
        plt.ylabel('Variance (degrees^2/frame^2)')
        plt.legend()
        plt.grid()
        plt.show(block=False)