# Given a yaml file of experiment sessions, plot the occupancy maps for each experiment in the session
import os
import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

from experimentStruct import ExperimentStruct

session_yaml = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/combined_sessions_byRat.yaml'
session_picked = 'Day1white'

# The yaml file is structured as:
# session1:
#   - /path/to/experiment1_timestamps.csv
#   - /path/to/experiment2_timestamps.csv
# session2:
#   - /path/to/experiment3_timestamps.csv
#   - /path/to/experiment4_timestamps.csv

# Keep all comparisons for occupancy maps
occupancy_maps = []

with open(session_yaml, 'r') as f:
    session_data = yaml.safe_load(f)
    # Plot data for the selected session
    for session_picked in ['Day1white', 'Day2white', 'Day3white', 'Day4white', 'Day5white', 'Day6white',
                           'Day1black', 'Day2black', 'Day3black', 'Day4black']:
        session_files = session_data[session_picked]

        # Each session has 6 files, so plot each occupancy map on a subplot
        # plt.subplots(3, 2, figsize=(5, 10))
        # plt.subplots_adjust(hspace=0.5)
        for i, timestampPath in enumerate(session_files):
            trial_data = ExperimentStruct(timestampPath)
            first, last = trial_data.firstLast_occupancy(sigma=0.3)

            if i == 0 or i == 3:
                test_distribution = last
            if i == 1 or i == 4:
                intr_distribution = last

            # Compute Wasserstein distance
            wasserstein_distance_test = stats.wasserstein_distance_nd(test_distribution, first)
            wasserstein_distance_sess = stats.wasserstein_distance_nd(last, first)
            if(i != 0 and i != 3):
                wasserstein_distance_intr = stats.wasserstein_distance_nd(intr_distribution, first)
            else:
                wasserstein_distance_intr = stats.wasserstein_distance_nd(last, first)

            # if i > 2:
            #     plt.subplot(3, 2, 2*i - 5)
            #     plt.imshow(first.T, cmap='hot', interpolation='nearest')
            #     plt.title(f'{trial_data.experimentTag}\nEMD from Train: {wasserstein_distance_test}\nEMD from Int: {wasserstein_distance_intr}')
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.subplot(3, 2, 2*i - 4)
            #     plt.imshow(last.T, cmap='hot', interpolation='nearest')
            #     plt.xticks([])
            #     plt.yticks([])
            #     plt.title(f'Last\nEMD Session: {wasserstein_distance_sess}')

            # Put more space between subplots
            
            # Store the experiment tag and wasserstein distances
            occupancy_maps.append((trial_data.experimentTag, wasserstein_distance_sess, wasserstein_distance_test, wasserstein_distance_intr))

    # plt.tight_layout()
    # plt.show(block=True)

# Convert occupancy_maps to a DataFrame and plot values 
occupancy_df = pd.DataFrame(occupancy_maps, columns=['ExperimentTag', 'Wasserstein_Session', 'Wasserstein_Test', 'Wasserstein_Interference'])
# Further split group by session type, given that the order repeats every 3 experiments using modulo
occupancy_df['SessionType'] = occupancy_df.index.map(lambda x: 'Train' if x % 3 == 0 else ('Interference' if x % 3 == 1 else 'Test'))

# Plot Wasserstein distances by session type, plotting in order 'Train', 'Interference', 'Test'
plt.figure(figsize=(8, 6))
for session_type, group_data in occupancy_df.groupby('SessionType'):
    plt.scatter([session_type]*len(group_data), group_data['Wasserstein_Session'], alpha=0.5, label=session_type)
    # Plot means and std devs
    mean_val = group_data['Wasserstein_Session'].mean()
    std_val = group_data['Wasserstein_Session'].std()
    plt.errorbar(session_type, mean_val, yerr=std_val, fmt='o', capsize=5)

plt.ylabel('Wasserstein Distance (a.u.)')
plt.title('Wasserstein Distances Within Session')
plt.show(block=False)

plt.figure(figsize=(8, 6))
for session_type, group_data in occupancy_df.groupby('SessionType'):
    plt.scatter([session_type]*len(group_data), group_data['Wasserstein_Test'], alpha=0.5, label=session_type)
    # Plot means and std devs
    mean_val = group_data['Wasserstein_Test'].mean()
    std_val = group_data['Wasserstein_Test'].std()
    plt.errorbar(session_type, mean_val, yerr=std_val, fmt='o', capsize=5)

plt.ylabel('Wasserstein Distance (a.u.)')
plt.title('Wasserstein Distances From Training Session')
plt.show(block=False)

plt.figure(figsize=(8, 6))
for session_type, group_data in occupancy_df.groupby('SessionType'):
    plt.scatter([session_type]*len(group_data), group_data['Wasserstein_Interference'], alpha=0.5, label=session_type) 
    # Plot means and std devs
    mean_val = group_data['Wasserstein_Interference'].mean(skipna=True)
    std_val = group_data['Wasserstein_Interference'].std(skipna=True)
    plt.errorbar(session_type, mean_val, yerr=std_val, fmt='o', capsize=5)

plt.ylabel('Wasserstein Distance (a.u.)')
plt.title('Wasserstein Distances From Interference Session')
plt.show(block=True)