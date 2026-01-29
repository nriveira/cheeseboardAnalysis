# Given a folder of behavior and timing data, concatenate to be able to do statistical analyses on the group.

import os
import matplotlib.pyplot as plt
import pandas as pd

from experimentStruct import ExperimentStruct

# Given a folder of timestamp files, compile distance traveled metrics to compare across groups
def distance_traveled_analysis(groupFolder):
    # For each _timestamps file, run the analysis plot and make a matrix of distance traveled values
    distance_traveled_data = []
    max_trials = 0

    for file in os.listdir(groupFolder):
        if file.endswith('_timestamps.csv'):
            timestampPath = os.path.join(groupFolder, file)
            trial_data = ExperimentStruct(timestampPath)
            dt_values = trial_data.find_distance_traveled()

            max_trials = max(max_trials, len(dt_values))
            # If fewer trials than max, append NaNs
            if len(dt_values) < max_trials:
                dt_values.extend([float('nan')] * (max_trials - len(dt_values)))

            distance_traveled_data.append({
                'experimentTag': trial_data.experimentTag,
                'distanceTraveled': dt_values
            })

    # Convert to DataFrame for easier analysis
    distance_traveled_data = pd.DataFrame(distance_traveled_data)

    # Make a matrix of distance traveled values
    distance_matrix = pd.DataFrame({
        row['experimentTag']: row['distanceTraveled'] for _, row in distance_traveled_data.iterrows()
    })

    # Find mean and standard error across rows
    mean_distance = distance_matrix.mean(axis=1)
    sem_distance = distance_matrix.sem(axis=1)
    return mean_distance, sem_distance, distance_matrix

# Load in data from each experimental condition
# White-Black-White 1 Hour
preInt1Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PreInterference1HourWhite'
int1HrBlack = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/InterferenceBlack1Hr'
postInt1Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PostInterference1HourWhite'

# Black-White-Black 1 Hour
preInt1HrBlack = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PreInterference1HourBlackEdited'
int1Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/InterferenceWhite1Hr'
postInt1HrBlack = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PostInterference1HourBlackEdited'

# White-Black-White 4 Hour
preInt4Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PreInterference4HourWhite'
int4HrBlack = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/InterferenceBlack4Hr'
postInt4Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PostInterference4HourWhite'

# Black-White-Black 4 Hour
preInt4HrBlack = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PreInterference4HourBlack'
int4Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/InterferenceWhite4Hr'
postInt4HrBlack = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PostInterference4HourBlackEdited'

# Make a function to plot the mean distance traveled with error bars and individual data points of a full session
def plot_distance_traveled(pre, interfere, post, color, offset, preIntOffset, postIntOffset, label, scatter=True):
    preMean, preSem, preMatrix = distance_traveled_analysis(pre)
    interfereMean, interfereSem, interfereMatrix = distance_traveled_analysis(interfere)
    postMean, postSem, postMatrix = distance_traveled_analysis(post)

    plt.errorbar(preMean.index+offset, preMean.values, yerr=preSem.values, color=color, label=label + ' Pre Interference')
    plt.errorbar(interfereMean.index+20+offset+preIntOffset, interfereMean.values, yerr=interfereSem.values, color=color)
    plt.errorbar(postMean.index+30+offset+preIntOffset+postIntOffset, postMean.values, yerr=postSem.values, color=color)
    if scatter:
        plt.scatter(preMatrix.index.repeat(preMatrix.shape[1])+offset, preMatrix.values.flatten(), color=color, alpha=0.3)
        plt.scatter(interfereMatrix.index.repeat(interfereMatrix.shape[1])+20+offset+preIntOffset, interfereMatrix.values.flatten(), color=color, alpha=0.3)
        plt.scatter(postMatrix.index.repeat(postMatrix.shape[1])+30+offset+preIntOffset+postIntOffset, postMatrix.values.flatten(), color=color, alpha=0.3)
plt.figure()
plt.subplot(2,1,1)
plot_distance_traveled(preInt1Hr, int1HrBlack, postInt1Hr, color='lime', offset=-0.15, preIntOffset=0, postIntOffset=10, label='1 Hour WBW', scatter=False)
plot_distance_traveled(preInt1HrBlack, int1Hr, postInt1HrBlack, color='blue', offset=-0.05, preIntOffset=0, postIntOffset=10, label='1 Hour BWB', scatter=False)
# Put the legend to the right outside of the plot
plt.legend(loc='right')
plt.ylim(0, 2200)
plt.xlabel('Trials')
# Hide x axis labels
plt.xticks([])
plt.ylabel('Distance Traveled (cm)')
plt.title('Mean Distance Traveled')

plt.subplot(2,1,2)
plot_distance_traveled(preInt4Hr, int4HrBlack, postInt4Hr, color='orange', offset=0.05, preIntOffset=10, postIntOffset=0, label='4 Hour WBW', scatter=False)
plot_distance_traveled(preInt4HrBlack, int4Hr, postInt4HrBlack, color='red', offset=0.15, preIntOffset=10, postIntOffset=0, label='4 Hour BWB', scatter=False)
plt.legend(loc='right')
plt.ylim(0, 2200)
plt.xlabel('Trials')
plt.xticks([])
plt.ylabel('Distance Traveled (cm)')
plt.tight_layout()

plt.show(block=True)
