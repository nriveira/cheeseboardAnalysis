# Given a csv with timestamps of each frame with annotations 
# Create a new csv with the data split by trial

import pandas as pd
import time

input_csv = r"C:\DATA\NICK Cheeseboard\Processing Copies\ExperimentVideo_2025-08-14_1426_timestamps.csv"

def split_timestamps_by_trial(input_csv, mode='Duration'):
    df = pd.read_csv(input_csv, header=None, names=['UnixTime', 'Monotonic', 'Event']).copy()

    # Only keep rows that do not have 0 in the Event column
    trial_df = df[df['Event'] != 0]

    # Convert the first column to time
    trial_df['Time'] = pd.to_datetime(trial_df['UnixTime'].astype('int64'), unit='ns')
    # Convert the second column to duration, and subtract the first value from all
    trial_df['Duration'] = pd.to_timedelta(trial_df['Monotonic'].astype('int64'), unit='ns')
    trial_df['Duration'] = trial_df['Duration'] - trial_df['Duration'].iloc[0]

    # # Reset the index
    trial_df = trial_df.reset_index(drop=True)

    # Go through the trial_df row by row. 
    # For each row, if the value is a 1, start a new row and add the duration in the first column
    # Otherwise, put the duration value in the row corresponding the the event number.
    formatted_trial_df = pd.DataFrame(columns=['Block',
                                               'Start_Unix', 'R1_Unix', 'R2_Unix', 'R3_Unix', 'End_Unix', 
                                               'Start_Mono', 'R1_Mono', 'R2_Mono', 'R3_Mono', 'End_Mono', 
                                               'Start_Time', 'R1_Time', 'R2_Time', 'R3_Time', 'End_Time', 
                                               'Start_Duration', 'R1_Duration', 'R2_Duration', 'R3_Duration', 'End_Duration'])
    loc = 0
    block = 1
    for i in range(len(trial_df)):
        if trial_df.iloc[i]['Event'] == 1:
            formatted_trial_df.loc[loc, 'Block'] = block // 10
            block += 1
            formatted_trial_df.loc[loc, 'Start_Unix'] = trial_df.iloc[i]['UnixTime']
            formatted_trial_df.loc[loc, 'Start_Mono'] = trial_df.iloc[i]['Monotonic']
            formatted_trial_df.loc[loc, 'Start_Time'] = trial_df.iloc[i]['Time']
            formatted_trial_df.loc[loc, 'Start_Duration'] = trial_df.iloc[i]['Duration']
        elif trial_df.iloc[i]['Event'] == 2:
            formatted_trial_df.loc[loc, 'R1_Unix'] = trial_df.iloc[i]['UnixTime']
            formatted_trial_df.loc[loc, 'R1_Mono'] = trial_df.iloc[i]['Monotonic']
            formatted_trial_df.loc[loc, 'R1_Time'] = trial_df.iloc[i]['Time']
            formatted_trial_df.loc[loc, 'R1_Duration'] = trial_df.iloc[i]['Duration']
        elif trial_df.iloc[i]['Event'] == 3:
            formatted_trial_df.loc[loc, 'R2_Unix'] = trial_df.iloc[i]['UnixTime']
            formatted_trial_df.loc[loc, 'R2_Mono'] = trial_df.iloc[i]['Monotonic']
            formatted_trial_df.loc[loc, 'R2_Time'] = trial_df.iloc[i]['Time']
            formatted_trial_df.loc[loc, 'R2_Duration'] = trial_df.iloc[i]['Duration']
        elif trial_df.iloc[i]['Event'] == 4:
            formatted_trial_df.loc[loc, 'R3_Unix'] = trial_df.iloc[i]['UnixTime']
            formatted_trial_df.loc[loc, 'R3_Mono'] = trial_df.iloc[i]['Monotonic']
            formatted_trial_df.loc[loc, 'R3_Time'] = trial_df.iloc[i]['Time']
            formatted_trial_df.loc[loc, 'R3_Duration'] = trial_df.iloc[i]['Duration']
        elif trial_df.iloc[i]['Event'] == 5:
            formatted_trial_df.loc[loc, 'End_Unix'] = trial_df.iloc[i]['UnixTime']
            formatted_trial_df.loc[loc, 'End_Mono'] = trial_df.iloc[i]['Monotonic']
            formatted_trial_df.loc[loc, 'End_Time'] = trial_df.iloc[i]['Time']
            formatted_trial_df.loc[loc, 'End_Duration'] = trial_df.iloc[i]['Duration']
            loc += 1

    return formatted_trial_df

def compute_trial_data(formatted_trial_df):
    # Compute metrics per trial (row):
    # Time to first reward, second reward, and third reward.
    # Create a new DataFrame to hold the computed metrics
    trial_durations = pd.DataFrame(columns=['Block', 'Trial Length', 'R1_Time', 'R2_Time', 'R3_Time', 'After_R3'])
    trial_durations['Block'] = formatted_trial_df['Block']
    trial_durations['Trial Length'] = (formatted_trial_df['End_Time'] - formatted_trial_df['Start_Time']).astype('timedelta64[s]')
    trial_durations['R1_Time'] = (formatted_trial_df['R1_Time'] - formatted_trial_df['Start_Time']).astype('timedelta64[s]')
    trial_durations['R2_Time'] = (formatted_trial_df['R2_Time'] - formatted_trial_df['Start_Time']).astype('timedelta64[s]')
    trial_durations['R3_Time'] = (formatted_trial_df['R3_Time'] - formatted_trial_df['Start_Time']).astype('timedelta64[s]')
    trial_durations['After_R3'] = (formatted_trial_df['End_Time'] - formatted_trial_df['R3_Time']).astype('timedelta64[s]')

    trial_durations['First Reward'] = None
    trial_durations['Last Reward'] = None

    for i in range(len(trial_durations)):
        trial_durations.loc[i, 'First Reward'] = min(trial_durations.loc[i, 'R1_Time'].total_seconds(), trial_durations.loc[i, 'R2_Time'].total_seconds(), trial_durations.loc[i, 'R3_Time'].total_seconds())
        trial_durations.loc[i, 'Last Reward'] = max(trial_durations.loc[i, 'R1_Time'].total_seconds(), trial_durations.loc[i, 'R2_Time'].total_seconds(), trial_durations.loc[i, 'R3_Time'].total_seconds())

    return trial_durations

split_by_trial_df = split_timestamps_by_trial(input_csv)
trial_data = compute_trial_data(split_by_trial_df)

print(trial_data)

# Plot the data per block
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
colors = ['blue', 'orange']
for block in trial_data['Block'].unique():
    block_data = trial_data[trial_data['Block'] == block]
    plt.plot(block_data['First Reward'], label='First Reward', color=colors[0])
    plt.plot(block_data['Last Reward'], label='Last Reward', color=colors[1])
plt.xlabel('Trial')
plt.ylabel('Time (s)')
plt.title(f'Trial Durations')
plt.legend()
plt.show()