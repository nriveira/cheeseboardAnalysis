# Given a folder of behavior and timing data, concatenate to be able to do statistical analyses on the group.
import os
import matplotlib.pyplot as plt
import pandas as pd

from experimentStruct import ExperimentStruct

# Given a folder of timestamp files, compile distance traveled metrics to compare across groups
def occupancy_analysis(groupFolder):
    # For each _timestamps file, run the analysis plot and make a matrix of occupancy values
    occupancy_data = []
    max_trials = 0

    for file in os.listdir(groupFolder):
        if file.endswith('_timestamps.csv'):
            timestampPath = os.path.join(groupFolder, file)
            trial_data = ExperimentStruct(timestampPath)
            trial_data.plot_occupancy_map()


preInt1Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/SEPARATE/PreInterference1HourWhite'
int1HrBlack = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/SEPARATE/InterferenceBlack1Hr'
postInt1Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/SEPARATE/PostInterference1HourWhite'

occupancy_analysis(int1HrBlack)