from convertTimestamp import Timestamp
from matplotlib import pyplot as plt
import pandas as pd

class Pathway():
    def plot_pathways(timestamp_file, tracking_part = 'nose1'):
        trials = Timestamp.convert_timestamp(timestamp_file)

        sleap_file = Timestamp.findSLEAPFile(timestamp_file)
        print("Found matching SLEAP file")

        sleap_data = pd.read_csv(sleap_file)

        plt.figure(figsize=(15, 10))
        for trial in trials:
            trial_number = trial['trial_number']
            start_idx = trial['start_idx']
            end_idx = trial['end_idx']

            trial_sleap_data = sleap_data[(sleap_data['frame_idx'] >= start_idx) & (sleap_data['frame_idx'] <= end_idx)]
            bodypart_x = f'{tracking_part}.x'
            bodypart_y = f'{tracking_part}.y'
            plt.subplot(4, 5, trial_number)

            # Filter the data to remove jumps by using rolling median of 5 points
            tracking_x = trial_sleap_data[bodypart_x].values
            tracking_y = trial_sleap_data[bodypart_y].values

            tracking_x = pd.Series(tracking_x).rolling(window=5, center=True).median().fillna(method='bfill').fillna(method='ffill').values
            tracking_y = pd.Series(tracking_y).rolling(window=5, center=True).median().fillna(method='bfill').fillna(method='ffill').values

            plt.plot(tracking_x, tracking_y, label=f'Trial {trial_number}')
            plt.title(f'Trial {trial_number}')
        plt.tight_layout()
        plt.show(block=True)