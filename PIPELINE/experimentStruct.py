# Code for parsing cheeseboard timestamp CSV files into structured trial data.
# Also must consider that SLEAP only reports frames that it successfully tracked,
# so there may be missing frames in the SLEAP data that need to be accounted for
# when linking timestamp data to SLEAP tracking data.

# Given an input timestamp file, organize the data using created classes to structure the data
# by structuring the timing table into a trial structure for alignment with SLEAP data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import re
import cv2
from datetime import datetime

from holographicBoard import HolographicBoard

class ExperimentStruct:
    """ 
    Organization of timestamp data into experimental block 
    """
    def __init__(self, timestamp_file, board_file=None):
        # 0) Basic experiment information
        self.occupancy_bin_size = 20  # Default bin size for occupancy maps
        # Recognize experiment date from metadata in file name: 
        self.experimentTag = self.extract_experiment_id(timestamp_file)
        # Remove the preceeding 'ExerimentVideo_' for cleaner display
        self.experimentTag = self.experimentTag.replace('ExperimentVideo_', '') if self.experimentTag else None
        self.experimentDatetime = datetime.strptime(self.experimentTag, '%Y-%m-%d_%H%M') if self.experimentTag else None

        # 1) Gather all data associated with experiment block
        self.timestamp_file = timestamp_file
        self.sleap_file = self.findSLEAPFile(timestamp_file)
        self.video_file = self.findVideoFile(timestamp_file)
        if not self.video_file:
            print(f"No Video for {self.experimentTag}")
        self.experiment_times = pd.read_csv(self.timestamp_file, header=None)

        # 2) Extract the first frame from the video file to use as a background image for plotting
        if self.video_file:
            video_capture = cv2.VideoCapture(self.video_file)
            ret, frame = video_capture.read()
            if ret:
                self.background_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB for plotting
            else:
                print(f"Warning: Could not read video file: {self.video_file}")
                self.background_image = None
            video_capture.release()
        else:
            self.background_image = None

        # 3) Run preprocessing functions to create a structure for analysis
        self.sleap_data_preprocessing()
        self.convert_trial_structure()
        self.board = HolographicBoard(self.experimentTag, self.background_image, board_file)

        self.ref_dist = self.estimate_distance()
        self.calculate_distance_traveled()

        # 3) Preliminary data processing (Add to same structure for easier access during analysis)
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

    def sleap_data_preprocessing(self, jump_threshold=30, smoothing_window=11):
        """ 
        Preprocess SLEAP data to match missing frames 
            and smooth tracking data to account for tracking errors and missing frames.
        Args:
            jump_threshold (float): Threshold for detecting tracking jumps (in pixels). Frames with jumps greater than this threshold will be set to NaN before smoothing.
            smoothing_window (int): Window size for rolling median smoothing (in frames).
        Returns:
            None: The processed SLEAP data will be stored in self.sleap_data
        """
        sleap_data = pd.read_csv(self.sleap_file)
        # Drop the tail and tail_end columns 
        sleap_data = sleap_data.drop(columns=[col for col in sleap_data.columns if 'tail1' in col or 'tailend' in col])

        # Make an empty DataFrame with all of the frame indices from the timestamp file
        timestamp_data = self.experiment_times.index.to_series().reset_index(drop=True)
        # Make an empty DataFrame with all frame indices
        all_frames = pd.DataFrame({'frame_idx': timestamp_data})
        # Merge SLEAP data with all frames to fill in missing frames
        merged_data = pd.merge(all_frames, sleap_data, on='frame_idx', how='left')

        # Compute the centroid of the tracked body parts for each frame
        body_parts = [col.split('.')[0] for col in merged_data.columns if '.x' in col or '.y' in col]
        body_parts = list(set(body_parts))  # Remove duplicates
        # Before calculating the centroid, take the rolling median of each body part's x and y coordinates to smooth the data
        for bp in body_parts:
            for axis in ['x', 'y']:
                col_name = f'{bp}.{axis}'
                merged_data[col_name] = merged_data[col_name].rolling(window=smoothing_window, min_periods=1, center=True).median()

        merged_data['centroid.x'] = merged_data[[f'{bp}.x' for bp in body_parts]].mean(axis=1)
        merged_data['centroid.y'] = merged_data[[f'{bp}.y' for bp in body_parts]].mean(axis=1)

        # Before smoothing, estimate number of jump points per column
        jump_points = 0
        # DATA SMOOTHING
        # for col in merged_data.columns:
        #     if '.x' in col or '.y' in col:
        #         jump_points = (merged_data[col].diff().abs() > jump_threshold).sum()
        #         # First, find any frames with jump values that are greater than the jump_threshold from the previous frame, and set those to NaN (these are likely tracking errors)
        #         merged_data[col] = merged_data[col].mask(merged_data[col].diff().abs() > jump_threshold)

        #         # Smooth all columns with '.x' or '.y' in the name using a rolling median with a window of smoothing_window frames, to account for missing frames and tracking noise
        #         merged_data[col] = merged_data[col].rolling(window=smoothing_window, min_periods=1, center=True).median()
        
        self.jump_points = jump_points  # Store the number of jump points for reference in analysis
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
        
        return None
    
    def findVideoFile(self, timestamp_file):
        """
        Find video file using experiment ID extraction for robust matching
        """
        # Extract experiment ID from timestamp filename
        experiment_id = self.extract_experiment_id(timestamp_file)
        
        # Search for video file containing the experiment ID
        base_dir = os.path.dirname(timestamp_file)
        for file in os.listdir(base_dir):
            if experiment_id in file and (file.endswith('.mp4')):
                return os.path.join(base_dir, file)
        
        return None
    
    def estimate_distance(self):
        """
        Estimate distance between two body parts across all frames to estimate 
        nose to neck distance, which can be used as a reference distance for spatial analyses (e.g., defining proximity to reward location).
        """
        
        nose_x = self.sleap_data['nose1.x'].values
        nose_y = self.sleap_data['nose1.y'].values
        neck_x = self.sleap_data['neck1.x'].values
        neck_y = self.sleap_data['neck1.y'].values

        delta_x = nose_x - neck_x
        delta_y = nose_y - neck_y

        distances = np.sqrt(delta_x**2 + delta_y**2)
        estimated_distance = np.nanmedian(distances)  # Use median to avoid outliers
        return estimated_distance

    # FIRST PASS VISUALIZATION FUNCTIONS/PLOTS
    def plot_single_trial(self, trial_num):
        """ Plot the head direction and position for a single trial """
        trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
        # Convert from Series to scalar values for indexing
        start_idx = int(trial_info['start_idx'].values[0])
        end_idx = int(trial_info['end_idx'].values[0])
        colors = ['red', 'blue', 'red']
        
        # Further split up the trial into start_idx, left_sb_idx, reward1_idx, reward2_idx, reward3_idx, end_idx to plot as different colors
        trial_segments = {
            'start': (start_idx, int(trial_info['left_sb_idx'].values[0])),
            'left_sb': (int(trial_info['left_sb_idx'].values[0]), int(trial_info['reward3_idx'].values[0])),
            # 'reward1': (int(trial_info['reward1_idx'].values[0]), int(trial_info['reward2_idx'].values[0])),
            # 'reward2': (int(trial_info['reward2_idx'].values[0]), int(trial_info['reward3_idx'].values[0])),
            'reward3': (int(trial_info['reward3_idx'].values[0]), end_idx)
        }

        # Use the background image from the video as the plot background for the trial pathway
        if self.background_image is not None:
            plt.imshow(self.background_image)
            # Remove the axes for cleaner visualization
            plt.axis('off')

        # Plot reward locations if they exist from HolographicBoard as LARGE dots
        if self.board.R1_xpixel is not None and self.board.R1_ypixel is not None:
            plt.scatter(self.board.R1_xpixel, self.board.R1_ypixel, color='red', s=100, label='Reward 1')
        if self.board.R2_xpixel is not None and self.board.R2_ypixel is not None:
            plt.scatter(self.board.R2_xpixel, self.board.R2_ypixel, color='green', s=100, label='Reward 2')
        if self.board.R3_xpixel is not None and self.board.R3_ypixel is not None:
            plt.scatter(self.board.R3_xpixel, self.board.R3_ypixel, color='blue', s=100, label='Reward 3')

        for segment, (seg_start, seg_end) in trial_segments.items():
            segment_data = self.sleap_data[(self.sleap_data['frame_idx'] >= seg_start) & (self.sleap_data['frame_idx'] < seg_end)]
            plt.plot(segment_data['centroid.x'], segment_data['centroid.y'], label=segment, color=colors.pop(0))
        
        plt.tight_layout()

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

        # Add the background image from the video as the plot background for the trial pathway
        if self.background_image is not None:
            plt.imshow(self.background_image)
            # Remove the axes for cleaner visualization
            plt.axis('off')

        for trial_num in self.trial_data['trial_num'].unique():
            trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
            if trial_info.empty:
                print(f"No data found for trial number {trial_num}.")
                continue

            left_sb_idx = int(trial_info['left_sb_idx'].values[0])
            end_idx = min(int(trial_info['end_idx'].values[0]), left_sb_idx + n_frames)
            
            trial_data = self.sleap_data[(self.sleap_data['frame_idx'] >= left_sb_idx) & (self.sleap_data['frame_idx'] < end_idx)]
            plt.plot(trial_data['centroid.x'], trial_data['centroid.y'], label=f'Trial {trial_num}')


    def plot_all_trials(self):
        """ 
        Plot the pathway for all trials, on separate subplots, with different colors for each trial segment 
        with 5 subplots per row, and as many rows as needed to fit all trials. Each trial should be plotted with the same color scheme for segments as in plot_single_trial.
        """  
        plt.figure(figsize=(20,15))

        plt.subplots_adjust(hspace=0.5)
        for trial_num in self.trial_data['trial_num'].unique():
            plt.subplot((len(self.trial_data) // 5) + 1, 5, trial_num + 1)
            self.plot_single_trial(trial_num)

        plt.suptitle(f'Pathway for {self.experimentTag}')
        plt.tight_layout()
    
    # VELOCITY AND DISTANCE FUNCTIONS
    def calculate_distance_traveled(self, tracking_point='centroid', normalize=None):
        """ 
        Calculate the distance traveled during each trial, using the specified tracking point data from SLEAP. 
        Only consider frames between leaving the start box and receiving the reward, to focus on the main trial behavior.
        """
        distances = []
        for trial_num in self.trial_data['trial_num'].unique():
            trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
            left_sb_idx = int(trial_info['left_sb_idx'].values[0])
            reward3_idx = int(trial_info['reward3_idx'].values[0])
            
            trial_data = self.sleap_data[(self.sleap_data['frame_idx'] >= left_sb_idx) & (self.sleap_data['frame_idx'] < reward3_idx)]
            
            # Calculate distance traveled as the sum of Euclidean distances between consecutive positions of the specified tracking point
            # While also accounting for any missing frames (which will have been smoothed in the preprocessing step)
            trial_data = trial_data.dropna(subset=[f'{tracking_point}.x', f'{tracking_point}.y'])

            # print(f"found {len(trial_data)} frames for trial {trial_num}")

            x_diff = trial_data[f'{tracking_point}.x'].diff().fillna(0)
            y_diff = trial_data[f'{tracking_point}.y'].diff().fillna(0)

            # print(f"Max x_diff for trial {trial_num}: {x_diff.abs().max()}")

            # Remove any points where the jump between frames is greater than 30
            jump_threshold = 30
            x_diff = x_diff.mask(x_diff.abs() > jump_threshold)
            y_diff = y_diff.mask(y_diff.abs() > jump_threshold)

            # print(self.ref_dist)

            trial_distance = np.sqrt(x_diff**2 + y_diff**2).sum() / self.ref_dist  # Normalize by reference distance to get distance in units of body lengths
            distances.append((trial_num, trial_distance))

            # print(distances)
        # Normalize all distances by dividing by the minimum distance across a session
        if normalize is None:
            self.min_distance = min(dist for _, dist in distances)
            distances = [(trial_num, dist / self.min_distance) for trial_num, dist in distances]
        else:
            # Normalize by the normalize value 
            distances = [(trial_num, dist / normalize) for trial_num, dist in distances]

        self.distance_traveled = distances  # Store distances for potential use in other analyses
        return distances
    
    def get_distance_traveled(self, normalize=None):
        """ 
        Return the distance traveled during each trial
        """
        distances = self.calculate_distance_traveled(normalize=normalize)
        trial_nums, trial_distances = zip(*distances)
        
        return trial_distances

    def plot_distance_traveled(self, color='blue',label='Distance Traveled', normalize=None, trial_offset=0):
        """ 
        Plot the distance traveled during each trial using a scatter plot
        """
        distances = self.calculate_distance_traveled(normalize=normalize)
        if not distances:
            print("No distance data available for plotting.")
            return
        
        trial_nums, trial_distances = zip(*distances)
        trial_nums = [num + trial_offset for num in trial_nums]
        plt.scatter(trial_nums, trial_distances, color=color, label=label)
        
        return trial_distances

    def plot_velocity(self, trial_num):
        """ 
        Plot instantaneous velocity for a single trial
        """
        trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
        start_idx = int(trial_info['start_idx'].values[0])
        end_idx = int(trial_info['end_idx'].values[0])
        
        # Further split up the trial into start_idx, left_sb_idx, reward1_idx, reward2_idx, reward3_idx, end_idx to plot as different colors
        trial_segments = {
            'start': (start_idx, int(trial_info['left_sb_idx'].values[0])),
            'left_sb': (int(trial_info['left_sb_idx'].values[0]), int(trial_info['reward3_idx'].values[0])),
            # 'reward1': (int(trial_info['reward1_idx'].values[0]), int(trial_info['reward2_idx'].values[0])),
            # 'reward2': (int(trial_info['reward2_idx'].values[0]), int(trial_info['reward3_idx'].values[0])),
            'reward3': (int(trial_info['reward3_idx'].values[0]), end_idx)
        }
        
        # Only plot left_sb to reward3 segment for velocity, since this is the main trial behavior
        seg_start, seg_end = trial_segments['left_sb']
        trial_data = self.sleap_data[(self.sleap_data['frame_idx'] >= seg_start) & (self.sleap_data['frame_idx'] < seg_end)]
        trial_data = trial_data.dropna(subset=['centroid.x', 'centroid.y'])
        x_diff = trial_data['centroid.x'].diff().fillna(0)
        y_diff = trial_data['centroid.y'].diff().fillna(0)
        velocity = np.sqrt(x_diff**2 + y_diff**2) * 30 / self.ref_dist  # Convert to body lengths per second assuming 30 fps
        plt.plot(trial_data['frame_idx'], velocity, color='blue')
        plt.title(f'JP: {(velocity > 30).sum()}')
        return (velocity > 30).sum()
        # print("Number of points over 30cm/s:", (velocity > 30).sum())  # Print number of points where velocity exceeds 30 cm/s for reference

    def plot_velocity_all_trials(self):
        """ 
        Plot instantaneous velocity for all trials on the same plot, with different colors for each trial.
        """
        plt.figure(figsize=(20,10))
        plt.subplots_adjust(hspace=0.5)
        jump_count = 0
        for trial_num in self.trial_data['trial_num'].unique():
            plt.subplot((len(self.trial_data) // 5) + 1, 5, trial_num + 1)
            jump_count += self.plot_velocity(trial_num)
        plt.suptitle(f'Velocity for {self.experimentTag}: {jump_count} Jumps')

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
