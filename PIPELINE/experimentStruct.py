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
        self.occupancy_bin_size = 50  # Default bin size for occupancy maps
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
        # Read in SLEAP data
        self.sleap_data_preprocessing()
        # Convert timestamp data to trial structure
        self.convert_trial_structure()
        # Use input data to align reward locations
        self.board = HolographicBoard(self.experimentTag, self.background_image, board_file)

        # 3) Preliminary data processing (Add to same structure for easier access during analysis)
        self.ref_dist = self.estimate_distance()
        self.calculate_distance_traveled()
        self.find_head_direction()
        self.find_putative_headscan()
        
    # PREPROCESSING HELPER FUNCTIONS
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

    # Preprocessing SLEAP data
    def sleap_data_preprocessing(self, jump_threshold=30, smoothing_window=5):
        """ 
        Preprocess SLEAP data:
            1) Add missing frames to match timestamp indices
            2) Remove noisy tracking points (tail data)
            2) Smooth tracking data 
                a) Removing tracking jumps by threshold
                b) Applying a rolling median filter to smooth data

        Args:
            jump_threshold (float): Threshold for detecting tracking jumps (in pixels). Frames with jumps greater than this threshold will be set to NaN before smoothing.
            smoothing_window (int): Window size for rolling median smoothing (in frames).
        Returns:
            None: The processed SLEAP data will be stored in self.sleap_data
        """
        sleap_data = pd.read_csv(self.sleap_file)
        # Drop the tail and tail_end columns 
        sleap_data = sleap_data.drop(columns=[col for col in sleap_data.columns if 'tail1' in col or 'tailend' in col])

        # 1) Add missing frames to match timestamp indices 
        timestamp_data = self.experiment_times.index.to_series().reset_index(drop=True)
        all_frames = pd.DataFrame({'frame_idx': timestamp_data})
        merged_data = pd.merge(all_frames, sleap_data, on='frame_idx', how='left')

        # 2) Smooth tracking data by removing tracking jumps by threshold
        # For each body part, calculate a rolling median, and removing any points with jumps above a threshold from the median
        body_parts = [col.split('.')[0] for col in merged_data.columns if '.x' in col or '.y' in col]
        body_parts = list(set(body_parts))
        for bp in body_parts:
            x_diff = merged_data[f'{bp}.x'].diff().fillna(jump_threshold + 1)
            y_diff = merged_data[f'{bp}.y'].diff().fillna(jump_threshold + 1)
            total_diff = np.sqrt(x_diff**2 + y_diff**2)

            # Remove any points where the jump between frames is greater than the threshold
            merged_data[f'{bp}.x'] = merged_data[f'{bp}.x'].mask(total_diff > jump_threshold)
            merged_data[f'{bp}.y'] = merged_data[f'{bp}.y'].mask(total_diff > jump_threshold)

            # 3) Smooth tracking data by applying a rolling median filter to smooth data
            merged_data[f'{bp}.x'] = merged_data[f'{bp}.x'].rolling(window=smoothing_window, center=True, min_periods=1).median()
            merged_data[f'{bp}.y'] = merged_data[f'{bp}.y'].rolling(window=smoothing_window, center=True, min_periods=1).median()

        # Compute the centroid of the tracked body parts for each frame
        merged_data['centroid.x'] = merged_data[[f'{bp}.x' for bp in body_parts]].mean(axis=1)
        merged_data['centroid.y'] = merged_data[[f'{bp}.y' for bp in body_parts]].mean(axis=1)

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

    # FIRST PASS VISUALIZATION PLOTTING FUNCTIONS
    def plot_single_trial(self, trial_num, bodypart='centroid'):
        """ 
        Plot the position of the rat for a single trial 
        """

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
            plt.plot(segment_data[f'{bodypart}.x'], segment_data[f'{bodypart}.y'], label=segment, color=colors.pop(0))
        
        plt.tight_layout()

    def plot_all_trials(self, bodypart='centroid'):
        """ 
        Plot the pathway for all trials, on separate subplots, with different colors for each trial segment 
        with 5 subplots per row, and as many rows as needed to fit all trials. Each trial should be plotted with the same color scheme for segments as in plot_single_trial.
        """  
        plt.figure(figsize=(20,15))

        plt.subplots_adjust(hspace=0.5)
        for trial_num in self.trial_data['trial_num'].unique():
            plt.subplot((len(self.trial_data) // 5) + 1, 5, trial_num + 1)
            self.plot_single_trial(trial_num, bodypart=bodypart)

        plt.suptitle(f'Pathway for {self.experimentTag}')
        plt.tight_layout()
    
    # CALCULATING DISTANCE TRAVELED
    def calculate_distance_traveled(self, tracking_point='centroid', normalize=None):
        """ 
        Calculate the distance traveled during each trial, using the specified tracking point data from SLEAP. 
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

            x_diff = trial_data[f'{tracking_point}.x'].diff().fillna(0)
            y_diff = trial_data[f'{tracking_point}.y'].diff().fillna(0)

            trial_distance = np.sqrt(x_diff**2 + y_diff**2).sum() / self.ref_dist  # Normalize by reference distance to get distance in units of body lengths
            distances.append((trial_num, trial_distance))

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
    
    # CALCULATING OCCUPANCY MAPS/SPATIAL INFORMATION
    def calculate_spatial_information(self, occupancy_map):
        """ 
        Calculate the spatial information of the occupancy map using Kernel Density Estimation
        """
        # Flatten the occupancy map and remove zero entries to avoid issues with log(0)
        occupancy_values = occupancy_map.flatten()
        occupancy_values = occupancy_values[occupancy_values > 0]

        if len(occupancy_values) == 0:
            return 0  # If there are no occupied bins, spatial information is zero

        # Calculate the probability of occupancy for each bin
        p_i = occupancy_values / np.sum(occupancy_values)

        # Calculate spatial information using the formula: I = sum(p_i * log2(p_i / p_mean))
        p_mean = np.mean(p_i)
        spatial_info = np.sum(p_i * np.log2(p_i / p_mean))

        return spatial_info

    def calculate_occupancy_map(self, trial_num, tracking_point='centroid'):
        """ 
        Calculate the occupancy map during each trial, using the specified tracking point data from SLEAP. 
        """
        trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
        left_sb_idx = int(trial_info['left_sb_idx'].values[0])
        reward3_idx = int(trial_info['reward3_idx'].values[0])
        
        trial_data = self.sleap_data[(self.sleap_data['frame_idx'] >= left_sb_idx) & (self.sleap_data['frame_idx'] < reward3_idx)]
        
        # Calculate occupancy map as a 2D histogram of the specified tracking point's x and y positions during the trial
        x_edges = np.arange(0, self.background_image.shape[1] + self.occupancy_bin_size, self.occupancy_bin_size)
        y_edges = np.arange(0, self.background_image.shape[0] + self.occupancy_bin_size, self.occupancy_bin_size)

        occupancy_map, _, _ = np.histogram2d(trial_data[f'{tracking_point}.x'], trial_data[f'{tracking_point}.y'], bins=[x_edges, y_edges], density=True)
        # Also calculate the spatial information of the occupancy map and return the value
        spatial_info = self.calculate_spatial_information(occupancy_map)

        return occupancy_map, spatial_info
    
    def plot_occupancy_map(self, trial_num, tracking_point='centroid'):
        """ 
        Plot the occupancy map during each trial as a heatmap
        """
        occupancy_map, spatial_info = self.calculate_occupancy_map(trial_num, tracking_point=tracking_point)
        plt.imshow(occupancy_map.T, origin='lower', cmap='hot', extent=[0, self.background_image.shape[1], 0, self.background_image.shape[0]])
        plt.colorbar(label='Occupancy Probability')
        plt.title(f'Occupancy Map for Trial {trial_num} (Spatial Info: {spatial_info:.2f})')

    def plot_all_occupancy_maps(self, tracking_point='centroid'):
        """ 
        Plot the occupancy map during each trial as a heatmap, with all trials on the same plot for comparison
        """
        plt.figure(figsize=(20,15))
        plt.subplots_adjust(hspace=0.5)
        for trial_num in self.trial_data['trial_num'].unique():
            plt.subplot((len(self.trial_data) // 5) + 1, 5, trial_num + 1)
            self.plot_occupancy_map(trial_num, tracking_point=tracking_point)
            plt.title(f'Trial {trial_num} Spatial Info: {self.calculate_spatial_information(self.calculate_occupancy_map(trial_num, tracking_point=tracking_point)[0]):.2f}')
        plt.suptitle(f'Occupancy Maps for {self.experimentTag}')

    def calculate_mutual_information(self, occupancy_map1, occupancy_map2):
        """ 
        Calculate the mutual information between two occupancy maps using the formula: I(X;Y) = sum(p(x,y) * log2(p(x,y) / (p(x)*p(y))))
        """
        # Flatten the occupancy maps
        p_xy = (occupancy_map1.flatten() * occupancy_map2.flatten()) + 1e-10  # Add a small constant to avoid log(0)

        p_x = occupancy_map1.flatten() + 1e-10  # Add a small constant to avoid division by zero
        p_y = occupancy_map2.flatten() + 1e-10  # Add a small constant to avoid division by zero

        # Calculate mutual information
        mutual_info = np.sum(p_xy * np.log2(p_xy / (p_x * p_y)))

        return mutual_info
    
    def plot_mutual_information(self, tracking_point='centroid'):
        """ 
        Plot the mutual information comparing each trial to every other trial, then creating a heatmap of the mutual information values for a full session\
        """
        trial_nums = self.trial_data['trial_num'].unique()
        mutual_info_matrix = np.zeros((len(trial_nums), len(trial_nums)))

        for i, trial_num1 in enumerate(trial_nums):
            occupancy_map1, _ = self.calculate_occupancy_map(trial_num1, tracking_point=tracking_point)
            for j, trial_num2 in enumerate(trial_nums):
                occupancy_map2, _ = self.calculate_occupancy_map(trial_num2, tracking_point=tracking_point)
                mutual_info_matrix[i, j] = self.calculate_mutual_information(occupancy_map1, occupancy_map2)

        # plt.imshow(mutual_info_matrix, origin='lower', cmap='viridis')
        # plt.colorbar(label='Mutual Information')
        # plt.title(f'Mutual Information Between Trials for {self.experimentTag}')
        # plt.xlabel('Trial Number')
        # plt.ylabel('Trial Number')

        # Only plot the last trial compared to all previous trials to see how the rat's behavior changes across the session
        plt.figure(figsize=(10,5))
        plt.plot(trial_nums[:], mutual_info_matrix[-1,:], marker='o')

        # Also plot the diagonal of the mutual infromation matrix as a second plot
        plt.figure(figsize=(10,5))
        plt.plot(trial_nums[:], mutual_info_matrix.diagonal(), marker='o')

    # DISTRIBUTION OF ANGLES
    def calculate_angle_distribution(self, trial_num, tracking_point='centroid'):
        """ 
        Calculate the distribution of angles between consecutive position vectors during a trial, which can be used to assess the variability of the rat's movement patterns during a trial. 
        """
        trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
        left_sb_idx = int(trial_info['left_sb_idx'].values[0])
        reward3_idx = int(trial_info['reward3_idx'].values[0])
        
        trial_data = self.sleap_data[(self.sleap_data['frame_idx'] >= left_sb_idx) & (self.sleap_data['frame_idx'] < reward3_idx)]
        
        # Calculate angles between consecutive position vectors of the specified tracking point
        x_diff = trial_data[f'{tracking_point}.x'].diff().fillna(0)
        y_diff = trial_data[f'{tracking_point}.y'].diff().fillna(0)

        # Sort the angle distribution into spatial bins to make a matrix of angles across the board
        spatial_bins_x = np.arange(0, self.background_image.shape[1] + self.occupancy_bin_size, self.occupancy_bin_size)
        spatial_bins_y = np.arange(0, self.background_image.shape[0] + self.occupancy_bin_size, self.occupancy_bin_size)

        angle_distribution = np.zeros((len(spatial_bins_x)-1, len(spatial_bins_y)-1, 12))
        for i in range(1, len(trial_data)-1):
            x_bin = np.digitize(x_diff.iloc[i], spatial_bins_x) - 1
            y_bin = np.digitize(y_diff.iloc[i], spatial_bins_y) - 1
            angle = np.arctan2(y_diff.iloc[i], x_diff.iloc[i]) * (180 / np.pi)  # Convert to degrees
            # If the angle is negative, add 360 to make it positive (0 to 360 degrees)
            if angle <= 0:
                angle += 360
            while angle >= 360:
                angle -= 360

            angle_bin = int(angle // 30)  # Bin angles into 30 degree bins
            angle_distribution[x_bin, y_bin, angle_bin] += 1

        # Normalize the angle distribution by the total number of angles
        angle_distribution /= np.sum(angle_distribution)

        return angle_distribution

    # CALCULATING VELOCITY
    def calculate_velocity(self, trial_num, tracking_point='centroid'):
        """
        Calculate the instantaneous velocity during each trial, using the specified tracking point data from SLEAP.
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

        x_diff = self.sleap_data[f'{tracking_point}.x'].diff().fillna(0)
        y_diff = self.sleap_data[f'{tracking_point}.y'].diff().fillna(0)
        
        # Only plot left_sb to reward3 segment for velocity, since this is the main trial behavior
        seg_start, seg_end = trial_segments['left_sb']
        trial_data = self.sleap_data[(self.sleap_data['frame_idx'] >= seg_start) & (self.sleap_data['frame_idx'] < seg_end)]
        velocity = np.sqrt(x_diff**2 + y_diff**2) * 30 / self.ref_dist  # Convert to body lengths per second assuming 30 fps
        return trial_data, velocity

    def plot_velocity(self, trial_num, bodypart='centroid'):
        """ 
        Plot instantaneous velocity for a single trial
        """
        trial_data, velocity = self.calculate_velocity(trial_num, tracking_point=bodypart)
        plt.plot(trial_data['frame_idx'], velocity, color='blue')
        plt.title(f'JP: {(velocity > 30).sum()}')

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

    # CALCULATING HEAD DIRECTION
    def find_head_direction(self):
        """ Using nose and neck positions, calculate head direction angles,
         accounting for wrap-around at 360 degrees """
        if self.sleap_data is None:
            return []
        
        a_x = self.sleap_data['nose1.x'].values
        a_y = self.sleap_data['nose1.y'].values
        b_x = self.sleap_data['neck1.x'].values
        b_y = self.sleap_data['neck1.y'].values
        c_x = self.sleap_data['tailstart1.x'].values
        c_y = self.sleap_data['tailstart1.y'].values

        # Find the angle between the vector from neck to nose and the vector from neck to tailstart, which gives the head direction relative to the body
        # This accounts for wrap-around at 360 degrees by using the arctan2 function
        deltaA_x = a_x - b_x
        deltaA_y = a_y - b_y
        deltaC_x = c_x - b_x
        deltaC_y = c_y - b_y

        cos_angle = (deltaA_x * deltaC_x + deltaA_y * deltaC_y) / (np.sqrt(deltaA_x**2 + deltaA_y**2) * np.sqrt(deltaC_x**2 + deltaC_y**2))
        angles = np.arccos(cos_angle) * (180 / np.pi)
        
        # Add head direction as a column of sleap data
        self.sleap_data['head_direction'] = angles

    def find_putative_headscan(self, frames=12):
        """
        Using head direction data, find putative headscanning events, defined as 
        periods of time where the head direction changes rapidly back and forth, 
        which can be identified by looking for periods where the head direction changes 
        by more than a certain threshold within a short time window. Additionally, it 
        must last for more than 0.4 seconds (12 frames at 30 fps)
        """
        head_direction = self.sleap_data['head_direction'].values  # Fill NaN values with forward fill to avoid issues with diff
        # Replace NaN values with the previous valid value to avoid issues with diff
        head_direction = pd.Series(head_direction).fillna(0).values
        head_direction_diff = np.abs(np.diff(head_direction))
        # Smooth the head direction difference using a rolling median to avoid noise
        self.head_direction_diff = pd.Series(head_direction_diff).rolling(window=frames, center=True, min_periods=1).mean().values

    def plot_headscans_single_trial(self, trial_num, threshold=1, frames=12):
        """
        Plot the trajectory of the rat, with overlayed head direction data during headscan events
        """
        trial_info = self.trial_data[self.trial_data['trial_num'] == trial_num]
        start_idx = int(trial_info['start_idx'].values[0])
        end_idx = int(trial_info['reward3_idx'].values[0])

        trial_data = self.sleap_data[(self.sleap_data['frame_idx'] >= start_idx) & (self.sleap_data['frame_idx'] < end_idx)]

        plt.figure(figsize=(10,10))
        plt.imshow(self.background_image)
        plt.plot(trial_data['centroid.x'], trial_data['centroid.y'], color='blue')

        # By going frame by frame, find periods where the head direction changes by more than threshold for at least 12 frames (0.4 seconds at 30 fps)
        # If these periods are found, plot the trajectory of the nose during these periods in a different color to indicate putative headscanning events
        head_velocity = self.head_direction_diff
        body_velocity = self.calculate_velocity(trial_num, tracking_point='centroid')[1]

        for i in range(1, len(head_velocity) - 1):
            if head_velocity[i] > threshold and body_velocity[i] <= 3:  # Threshold for rapid head direction change
                # Check if this rapid change is sustained on average for at least 12 frames
                if np.mean(head_velocity[i:i+frames]) > threshold*frames:
                    plt.plot(trial_data['nose1.x'].values[i:i+frames], trial_data['nose1.y'].values[i:i+frames], color='black', linewidth=2)
        
        plt.show()