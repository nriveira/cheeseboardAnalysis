from cheeseboardTimestamp import Timestamp
from matplotlib import pyplot as plt
import pandas as pd

class Duration():
    def plot_durations(timestamp_file):
        timestamp = Timestamp(timestamp_file).trials

        # Plot the trial durations
        durations = [trial['duration'].total_seconds() for trial in timestamp]

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
        plt.show(block=True)