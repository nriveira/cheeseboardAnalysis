# Code for parsing cheeseboard timestamp CSV files into structured trial data.
# Also must consider that SLEAP only reports frames that it successfully tracked,
# so there may be missing frames in the SLEAP data that need to be accounted for
# when linking timestamp data to SLEAP tracking data.

# Given an input timestamp file, organize the data using created classes to structure the data
# by structuring the timing table into a trial structure

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import re

class ExperimentStruct:
    """ 
    Organization of timestamp data into experimental block 
    """
    def __init__(self, timestamp_file):
        # 0) Basic experiment information
        self.experiment_type = None
        self.occupancy_bin_size = 20  # Default bin size for occupancy maps
        # Recognize experiment date from metadata in file name: 
        self.experimentTag = self.extract_experiment_id(timestamp_file)
        # Remove the preceeding 'ExerimentVideo_' for cleaner display
        self.experimentTag = self.experimentTag.replace('ExperimentVideo_', '') if self.experimentTag else None
        print(f"Experiment Tag: {self.experimentTag}")

        # 1) Gather all data associated with experiment block
        self.timestamp_file = timestamp_file
        self.sleap_file = self.findSLEAPFile(timestamp_file)
        self.experiment_times = pd.read_csv(self.timestamp_file, header=None)
        self.x_lims = (0, 720)  # Assuming 720x540 video resolution
        self.y_lims = (0, 540)

        # Run prepocessing functions to create a structure for analysis
        self.sleap_data_preprocessing()
        self.convert_trial_structure()

        # 2) Preliminary data processing (Add to same structure for easier access during analysis)
        self.find_head_direction()
        

    def extract_experiment_id(self, timestamp_file):
        """
        Extract the experiment identifier pattern from filename.
        
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

    def sleap_data_preprocessing(self, jump_threshold=30, smoothing_window=5):
        """ 
        Preprocess SLEAP data to match missing frames 
        """
        sleap_data = pd.read_csv(self.sleap_file)
        # Make an empty DataFrame with all of the frame indices from the timestamp file
        timestamp_data = self.experiment_times.index.to_series().reset_index(drop=True)
        # Make an empty DataFrame with all frame indices
        all_frames = pd.DataFrame({'frame_idx': timestamp_data})
        # Merge SLEAP data with all frames to fill in missing frames
        merged_data = pd.merge(all_frames, sleap_data, on='frame_idx', how='left')

        # DATA SMOOTHING
        for col in merged_data.columns:
            if '.x' in col or '.y' in col:
                # First, find any frames with jump values that are greater than the jump_threshold from the previous frame, and set those to NaN (these are likely tracking errors)
                merged_data[col] = merged_data[col].mask(merged_data[col].diff().abs() > jump_threshold)

                # Smooth all columns with '.x' or '.y' in the name using a rolling median with a window of smoothing_window frames, to account for missing frames and tracking noise
                merged_data[col] = merged_data[col].rolling(window=smoothing_window, min_periods=1, center=True).median()
        

        self.sleap_data = merged_data
            
    def convert_trial_structure(self):
        """ 
        Extracts all trial times to create a table. Experiment timeline is 
        1: Start Trial
        2: Left start box
        3: Reward 1
        4: Reward 2
        5: Reward 3
        6: End Trial

        This function creates rows that correspond to each trial and gives index times for each of these events, 
        which can be used to link to SLEAP data and other analyses.
        """ 
        # Create a DataFrame to hold trial information
        trial_data = pd.DataFrame(columns=['trial_num', 
                                           'start_time', 
                                           'start_idx',
                                           'left_sb_time',
                                           'left_sb_idx',
                                        #    'reward1_time',
                                        #    'reward1_idx',
                                        #    'reward2_time', 
                                        #    'reward2_idx',
                                           'reward3_time', 
                                           'reward3_idx',
                                           'end_time',
                                           'end_idx'])
        
        # Update the trial data table by iterating through the timestamp data and filling in the appropriate columns based on the state
        trial_num = 0
        for index, row in self.experiment_times.iterrows():
            unix_time = row[0]
            state = row[2] 

            if state == 1:
                # Add it to the trial data table
                trial_data.loc[trial_num, 'trial_num'] = trial_num
                trial_data.loc[trial_num, 'start_time'] = unix_time
                trial_data.loc[trial_num, 'start_idx'] = index
            elif state == 2:
                trial_data.loc[trial_num, 'left_sb_time'] = unix_time
                trial_data.loc[trial_num, 'left_sb_idx'] = index
            # elif state == 3:
            #     trial_data.loc[trial_num, 'reward1_time'] = unix_time
            #     trial_data.loc[trial_num, 'reward1_idx'] = index
            # elif state == 4:
            #     trial_data.loc[trial_num, 'reward2_time'] = unix_time
            #     trial_data.loc[trial_num, 'reward2_idx'] = index
            elif state == 5:
                trial_data.loc[trial_num, 'reward3_time'] = unix_time
                trial_data.loc[trial_num, 'reward3_idx'] = index
            elif state == 6:
                trial_data.loc[trial_num, 'end_time'] = unix_time
                trial_data.loc[trial_num, 'end_idx'] = index
                trial_num += 1  

        self.trial_data = trial_data

    # HELPER FUNCTIONS
    def findSLEAPFile(self, timestamp_file):
        """
        Find SLEAP analysis file using experiment ID extraction for robust matching
        """
        # Extract experiment ID from timestamp filename
        experiment_id = self.extract_experiment_id(timestamp_file)
        
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

    # FIRST PASS VISUALIZATION FUNCTIONS/PLOTS
    def plot_single_trial(self, trial_num):
        """ Plot the head direction and position for a single trial """
        if self.sleap_data is None or self.trial_data is None:
            print("SLEAP data or trial data not available for plotting.")
            return
        
        trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
        if trial_info.empty:
            print(f"No data found for trial number {trial_num}.")
            return
        
        start_idx = int(trial_info['start_idx'])
        end_idx = int(trial_info['end_idx'])
        
        # Further split up the trial into start_idx, left_sb_idx, reward1_idx, reward2_idx, reward3_idx, end_idx to plot as different colors
        trial_segments = {
            'start': (start_idx, int(trial_info['left_sb_idx'])),
            'left_sb': (int(trial_info['left_sb_idx']), int(trial_info['reward3_idx'])),
            # 'reward1': (int(trial_info['reward1_idx']), int(trial_info['reward2_idx'])),
            # 'reward2': (int(trial_info['reward2_idx']), int(trial_info['reward3_idx'])),
            'reward3': (int(trial_info['reward3_idx']), end_idx)
        }
        for segment, (seg_start, seg_end) in trial_segments.items():
            segment_data = self.sleap_data[(self.sleap_data['frame_idx'] >= seg_start) & (self.sleap_data['frame_idx'] < seg_end)]
            plt.plot(segment_data['nose1.x'], segment_data['nose1.y'], label=segment)
        
        plt.tight_layout()
        plt.xlim(self.x_lims)
        plt.ylim(self.y_lims)

    def plot_n_seconds(self, n_seconds=5):
        """ 
        Plot the position for n seconds after leaving the start box 
        for all trials, on the same plot, with different colors for each trial.
        """
        if self.sleap_data is None or self.trial_data is None:
            print("SLEAP data or trial data not available for plotting.")
            return
        
        fps = 30  # Assuming 30 frames per second
        n_frames = n_seconds * fps

        for trial_num in self.trial_data['trial_num'].unique():
            trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
            if trial_info.empty:
                print(f"No data found for trial number {trial_num}.")
                continue

            left_sb_idx = int(trial_info['left_sb_idx'])
            end_idx = min(int(trial_info['end_idx']), left_sb_idx + n_frames)
            
            trial_data = self.sleap_data[(self.sleap_data['frame_idx'] >= left_sb_idx) & (self.sleap_data['frame_idx'] < end_idx)]
            plt.plot(trial_data['nose1.x'], trial_data['nose1.y'], label=f'Trial {trial_num}')

        plt.xlim(self.x_lims)
        plt.ylim(self.y_lims)
    

    def plot_all_trials(self):
        """ 
        Plot the pathway for all trials, on separate subplots, with different colors for each trial segment 
        with 5 subplots per row, and as many rows as needed to fit all trials. Each trial should be plotted with the same color scheme for segments as in plot_single_trial.
        """  
        plt.figure(figsize=(20,10))
        plt.subplots_adjust(hspace=0.5)
        for trial_num in self.trial_data['trial_num'].unique():
            plt.subplot((len(self.trial_data) // 5) + 1, 5, trial_num + 1)
            self.plot_single_trial(trial_num)

    # SPATIAL ANALYSIS FUNCTIONS
    def create_occupancy_map(self, num_trials=[0, 1, 2]):
        """ 
        Create an occupancy map of the arena based on the nose position data from SLEAP, 
        using a specified bin size for spatial bins. 
        Only consider frames between leaving the start box and receiving the reward, to focus on the main trial behavior.
        Plot only the first num_trials
        """
        if self.sleap_data is None:
            print("SLEAP data not available for occupancy map.")
            return
        
        full_occupancy = np.zeros((int((self.y_lims[1] - self.y_lims[0]) / self.occupancy_bin_size), 
                                   int((self.x_lims[1] - self.x_lims[0]) / self.occupancy_bin_size)))
        plt.figure(figsize=(10,10))
        for trial_num in self.trial_data['trial_num'].unique()[num_trials]:
            trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
            if trial_info.empty:
                print(f"No data found for trial number {trial_num}.")
                continue

            left_sb_idx = int(trial_info['left_sb_idx'])
            reward3_idx = int(trial_info['reward3_idx'])
            
            trial_data = self.sleap_data[(self.sleap_data['frame_idx'] >= left_sb_idx) & (self.sleap_data['frame_idx'] < reward3_idx)]
            # Add to the full occupancy map
            x_bins = np.arange(self.x_lims[0], self.x_lims[1] + self.occupancy_bin_size, self.occupancy_bin_size)
            y_bins = np.arange(self.y_lims[0], self.y_lims[1] + self.occupancy_bin_size, self.occupancy_bin_size)
            occupancy, _, _ = np.histogram2d(trial_data['nose1.x'], trial_data['nose1.y'], bins=[x_bins, y_bins])
            full_occupancy += occupancy

            plt.imshow(occupancy, origin='lower', extent=(self.x_lims[0], self.x_lims[1], self.y_lims[0], self.y_lims[1]), cmap='hot', alpha=0.5)
            plt.title(f'Trial {trial_num} Occupancy Map')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.colorbar(label='Occupancy Count')
            plt.show()
    
    # VELOCITY AND DISTANCE FUNCTIONS

    
    # HEAD DIRECTION FUNCTIONS
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

        head_directions = np.degrees(np.arctan2(delta_y, delta_x))
        
        # Add head direction as a column of sleap data
        self.sleap_data['head_direction'] = head_directions
