# Given a folder of behavior and timing data, concatenate to be able to do statistical analyses on the group.

import os
import matplotlib.pyplot as plt
import pandas as pd

from experimentStruct import ExperimentStruct

# Given a folder of timestamp files, compile distance traveled metrics to compare across groups
def distance_traveled_analysis(groupFolder):
    # For each _timestamps file, run the analysis plot and make a matrix of distance traveled values
    distance_traveled_data = []

    for file in os.listdir(groupFolder):
        if file.endswith('_timestamps.csv'):
            timestampPath = os.path.join(groupFolder, file)
            trial_data = ExperimentStruct(timestampPath)

            print(f"Processing experiment: {trial_data.experimentTag}")
            dt_values = trial_data.find_distance_traveled()
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

preInt1Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PreInterference1HourWhite'
preInt1HrMean, preInt1HrSem, preInt1HrMatrix = distance_traveled_analysis(preInt1Hr)
postInt1Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PostInterference1HourWhite'
postInt1HrMean, postInt1HrSem, postInt1HrMatrix = distance_traveled_analysis(postInt1Hr)
preInt4Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PreInterference4HourWhite'
preInt4HrMean, preInt4HrSem, preInt4HrMatrix = distance_traveled_analysis(preInt4Hr)
postInt4Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/PostInterference4HourWhite'
postInt4HrMean, postInt4HrSem, postInt4HrMatrix = distance_traveled_analysis(postInt4Hr)
interference = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/InterferenceWhite'
intfMean, intfSem, intfMatrix = distance_traveled_analysis(interference)

# Plot mean distance traveled with error bars
# plt.figure()
# plt.errorbar(preInt1HrMean.index, preInt1HrMean.values, yerr=preInt1HrSem.values, color='blue', label='Pre Interference 1 Hour')
# plt.errorbar(postInt1HrMean.index+20, postInt1HrMean.values, yerr=postInt1HrSem.values, color='blue', label='Post Interference 1 Hour')
# plt.errorbar(preInt4HrMean.index, preInt4HrMean.values, yerr=preInt4HrSem.values, color='red', label='Pre Interference 4 Hour')
# plt.errorbar(postInt4HrMean.index+20, postInt4HrMean.values, yerr=postInt4HrSem.values, color='red', label='Post Interference 4 Hour')
# plt.errorbar(intfMean.index, intfMean.values, yerr=intfSem.values, color='green', label='Interference')
# plt.title('Mean Distance Traveled')
# plt.legend()
# plt.xlabel('Time (frames)')
# plt.ylabel('Distance Traveled (cm)')
# plt.show(block=True)

# Along with plotting the mean values, also plot individual points for all experiments
# plt.figure()
# plt.errorbar(preInt1HrMean.index, preInt1HrMean.values, yerr=preInt1HrSem.values, color='blue', label='Pre Interference 1 Hour')
# # plt.scatter(preInt1HrMatrix.index.repeat(preInt1HrMatrix.shape[1])-0.1, preInt1HrMatrix.values.flatten(), color='blue', alpha=0.5)
# plt.errorbar(preInt4HrMean.index, preInt4HrMean.values, yerr=preInt4HrSem.values, color='red', label='Pre Interference 4 Hour')
# # plt.scatter(preInt4HrMatrix.index.repeat(preInt4HrMatrix.shape[1])+0.1, preInt4HrMatrix.values.flatten(), color='red', alpha=0.5)
# plt.errorbar(intfMean.index, intfMean.values, yerr=intfSem.values, color='green', label='Interference')
# # plt.scatter(intfMatrix.index.repeat(intfMatrix.shape[1]), intfMatrix.values.flatten(), color='green', alpha=0.5)
# plt.title('Mean Distance Traveled Pre Interference')
# plt.legend()
# plt.ylim(0, 2200)
# plt.xlabel('Time (frames)')
# plt.ylabel('Distance Traveled (cm)')
# plt.show(block=True)

# plt.figure()
# plt.errorbar(postInt1HrMean.index, postInt1HrMean.values, yerr=postInt1HrSem.values, color='blue', label='Post Interference 1 Hour')
# plt.scatter(postInt1HrMatrix.index.repeat(postInt1HrMatrix.shape[1])-0.1, postInt1HrMatrix.values.flatten(), color='blue', alpha=0.5)
# plt.errorbar(postInt4HrMean.index, postInt4HrMean.values, yerr=postInt4HrSem.values, color='red', label='Post Interference 4 Hour')
# plt.scatter(postInt4HrMatrix.index.repeat(postInt4HrMatrix.shape[1])+0.1, postInt4HrMatrix.values.flatten(), color='red', alpha=0.5)
# plt.errorbar(intfMean.index, intfMean.values, yerr=intfSem.values, color='green', label='Interference')
# plt.scatter(intfMatrix.index.repeat(intfMatrix.shape[1]), intfMatrix.values.flatten(), color='green', alpha=0.5)
# plt.title('Mean Distance Traveled Post Interference')
# plt.legend()
# plt.ylim(0, 1800)
# plt.xlabel('Time (frames)')
# plt.ylabel('Distance Traveled (cm)')
# plt.show(block=True)


# Also compare interference conditions
int1Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/InterferenceWhite1Hr'
int1HrMean, int1HrSem, int1HrMatrix = distance_traveled_analysis(int1Hr)
int4Hr = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/InterferenceWhite4Hr'
int4HrMean, int4HrSem, int4HrMatrix = distance_traveled_analysis(int4Hr)

plt.figure()
plt.errorbar(int1HrMean.index, int1HrMean.values, yerr=int1HrSem.values, color='purple', label='Interference 1 Hour')
plt.scatter(int1HrMatrix.index.repeat(int1HrMatrix.shape[1])-0.1, int1HrMatrix.values.flatten(), color='purple', alpha=0.5)
plt.errorbar(int4HrMean.index, int4HrMean.values, yerr=int4HrSem.values, color='orange', label='Interference 4 Hour')
plt.scatter(int4HrMatrix.index.repeat(int4HrMatrix.shape[1])+0.1, int4HrMatrix.values.flatten(), color='orange', alpha=0.5)
plt.title('Mean Distance Traveled During Interference')
plt.legend()
plt.ylim(0, 1800)
plt.xlabel('Time (frames)')
plt.ylabel('Distance Traveled (cm)')
plt.show(block=True)

plt.figure()
plt.errorbar(int1HrMean.index, int1HrMean.values, yerr=int1HrSem.values, color='purple', label='Interference 1 Hour')
# plt.scatter(int1HrMatrix.index.repeat(int1HrMatrix.shape[1])-0.1, int1HrMatrix.values.flatten(), color='purple', alpha=0.5)
plt.errorbar(int4HrMean.index, int4HrMean.values, yerr=int4HrSem.values, color='orange', label='Interference 4 Hour')
# plt.scatter(int4HrMatrix.index.repeat(int4HrMatrix.shape[1])+0.1, int4HrMatrix.values.flatten(), color='orange', alpha=0.5)
plt.title('Mean Distance Traveled During Interference')
plt.legend()
plt.ylim(0, 1800)
plt.xlabel('Time (frames)')
plt.ylabel('Distance Traveled (cm)')
plt.show(block=True)