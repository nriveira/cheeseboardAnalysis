# File for preliminary analysis of SLEAP data
from cheeseboardTimestamp import Timestamp
from matplotlib import pyplot as plt
import pandas as pd

class sleapAndTimestamp():
    def __init__(self, timestamp_file):
        # Import the timestamp file and find SLEAP data
        self.timestamp = Timestamp(timestamp_file)

        self.sleap_file = self.timestamp.sleap_file
        self.sleap_data = pd.read_csv(self.sleap_file)

        # Extract trial information
        self.trials = self.timestamp.trials
        self.durations = self.find_durations()
        self.velocities = self.find_velocity()

    def find_durations(self):
        return [trial['duration'].total_seconds() for trial in self.trials]

    def find_velocity(self, tracking_part='neck1', dt=30):
        # Calculate velocities at all points
        bodypart_x = f'{tracking_part}.x'
        bodypart_y = f'{tracking_part}.y'

        tracking_x = self.sleap_data[bodypart_x].values
        tracking_y = self.sleap_data[bodypart_y].values

        # Median filter to smooth tracking data
        tracking_x = pd.Series(tracking_x).rolling(window=5, center=True).median().values
        tracking_y = pd.Series(tracking_y).rolling(window=5, center=True).median().values

        # Calculate velocities
        dx = pd.Series(tracking_x).diff().fillna(0).values
        dy = pd.Series(tracking_y).diff().fillna(0).values

        speed = ((dx ** 2 + dy ** 2) ** 0.5) / dt
        return speed
        
    def plot_durations(self):
        durations = self.durations

        # Also do a moving average from the beginning
        window_size = 2
        moving_avg = pd.Series(durations).rolling(window=window_size).mean()

        plt.figure(figsize=(10, 5))
        plt.plot(durations, label='Trial Durations', marker='o')
        plt.plot(moving_avg, color='red', label=f'{window_size}-trial Moving Average')
        plt.legend()
        plt.title('Trial Durations Over Trials')
        plt.xlabel('Trial Number')
        plt.ylabel('Duration (seconds)')
        plt.grid()
        plt.show()

    def plot_pathways(self, tracking_part='neck1'):
        plt.figure(figsize=(15, 10))
        for trial in self.trials:
            trial_number = trial['trial_number']
            start_idx = trial['left_rest_box_idx'] if 'left_rest_box_idx' in trial else trial['start_idx']
            end_idx = trial['last_idx'] if 'last_idx' in trial else trial['end_idx']

            trial_sleap_data = self.sleap_data[(self.sleap_data['frame_idx'] >= start_idx) & (self.sleap_data['frame_idx'] <= end_idx)]
            bodypart_x = f'{tracking_part}.x'
            bodypart_y = f'{tracking_part}.y'

            plt.subplot(4, 5, trial_number)
            tracking_x = trial_sleap_data[bodypart_x].values
            tracking_y = trial_sleap_data[bodypart_y].values

            tracking_x = pd.Series(tracking_x).rolling(window=5, center=True).median().values
            tracking_y = pd.Series(tracking_y).rolling(window=5, center=True).median().values

            plt.plot(tracking_x, tracking_y, label=f'Trial {trial_number}')
            plt.title(f'Trial {trial_number} Pathway')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.show()

    def plot_velocity(self):
        velocities = self.velocities

        plt.figure(figsize=(15, 10))
        for trial in self.trials:
            start_idx = trial['left_rest_box_idx'] if 'left_rest_box_idx' in trial else trial['start_idx']
            end_idx = trial['last_idx'] if 'last_idx' in trial else trial['end_idx']

            trial_velocities = velocities[start_idx:end_idx + 1]

            plt.subplot(4, 5, trial['trial_number'])
            plt.plot(trial_velocities, label=f'Trial {trial["trial_number"]} Velocity')
            plt.title(f'Trial {trial["trial_number"]} Velocity Over Time')
            plt.xlabel('Frame Index')
            plt.ylabel('Velocity (pixels/frame)')
            plt.legend()
            plt.grid()
        plt.tight_layout()
        plt.show()