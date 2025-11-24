# Given a folder of csv files with trial data,
# this script will concatenate the data into a single csv file.
# It will also plot the mean and standard error per block

import pandas as pd
import os
import matplotlib.pyplot as plt

from legacy.process_timestamp import compute_trial_data

split_by_trial_folder = r"C:\DATA\NICK Cheeseboard\Experiment Trials"
output_csv = r"C:\DATA\NICK Cheeseboard\Experiment Data\combined_experiment_data.csv"

def combine_trials(input_folder, output_file):
    # List all CSV files in the input folder
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('split_by_trial.csv')]
    # Initialize an empty DataFrame to hold combined data
    combined_df = pd.DataFrame()
    print(csv_files)

    # Loop through each CSV file and append its data to the combined DataFrame
    block = 0
    for csv_file in csv_files:
        print(f"Processing {csv_file} as block {block}")
        file_path = os.path.join(input_folder, csv_file)
        df = pd.read_csv(file_path)
        # Change the 'Block' column to increasing numbers

        # Get the date from the filename between 'ExperimentVideo_' and '_split_by_trial'
        date_str = csv_file.split('ExperimentVideo_')[1].split('_split_by_trial')[0]
        df['Datetime'] = date_str
        block += 1

        # Compute trial data
        trial_data = compute_trial_data(df)
        trial_data['Block'] = block
        trial_data['Datetime'] = date_str
        trial_data['Trial'] = trial_data.index

        combined_df = pd.concat([combined_df, trial_data], ignore_index=True)
        # combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined data saved to {output_file}")

combine_trials(split_by_trial_folder, output_csv)

combined_df = pd.read_csv(output_csv)
# Use the combined DataFrame to plot mean and standard error per block
def plot_mean_and_se_R_Time(combined_df):
    # Group by block and compute mean and standard error per trial
    combined_df['R_Time'] = combined_df['R_Time'].astype('timedelta64[s]').dt.total_seconds()
    summary = combined_df.groupby('Trial').agg(
        R_Time_mean=('R_Time', 'mean'),
        R_Time_sem=('R_Time', 'sem')
    ).reset_index()
    # Plot mean and standard error, and individual points for all trials
    plt.figure(figsize=(10, 6))
    plt.errorbar(summary['Trial'], summary['R_Time_mean'], yerr=summary['R_Time_sem'], fmt='o', capsize=5)
    plt.title('Mean and Standard Error per Trial')
    plt.xlabel('Trial')
    plt.ylabel('R Time (seconds)')
    plt.xticks(summary['Trial'])
    plt.show()

    # Also plot individual points for all trials
    plt.figure(figsize=(10, 6))
    for block in combined_df['Block'].unique():
        block_data = combined_df[combined_df['Block'] == block]
        plt.scatter(block_data['Trial'], block_data['R_Time'].astype('timedelta64[s]').dt.total_seconds(), label=f'Block {block}', alpha=0.5)
    plt.title('Individual R Times per Trial')
    plt.xlabel('Trial')
    plt.ylabel('R Time (seconds)')
    # Make the legend outside the plot
    plt.legend(title='Block', bbox_to_anchor=(1.12, 1), loc='upper right')
    plt.show()

def plot_mean_and_se_First_Reward(combined_df):
    # Group by block and compute mean and standard error per trial
    combined_df['First Reward'] = combined_df['First Reward'].astype('timedelta64[s]').dt.total_seconds()

    summary = combined_df.groupby('Trial').agg(
        First_Reward_mean=('First Reward', 'mean'),
        First_Reward_sem=('First Reward', 'sem')
    ).reset_index()
    plt.figure(figsize=(10, 6))
    plt.errorbar(summary['Trial'], summary['First_Reward_mean'], yerr=summary['First_Reward_sem'], fmt='o', capsize=5)
    plt.title('Mean and Standard Error of First Reward per Trial')
    plt.xlabel('Trial')
    plt.ylabel('First Reward Time (seconds)')
    plt.xticks(summary['Trial'])
    plt.show()

    # Plot individual points for all trials
    plt.figure(figsize=(10, 6))
    for block in combined_df['Block'].unique():
        block_data = combined_df[combined_df['Block'] == block]
        plt.scatter(block_data['Trial'], block_data['First Reward'].astype('timedelta64[s]').dt.total_seconds(), label=f'Block {block}', alpha=0.5)
    plt.title('Individual First Reward Times per Trial')
    plt.xlabel('Trial')
    plt.ylabel('First Reward Time (seconds)')
    # Make the legend outside the plot
    plt.legend(title='Block', bbox_to_anchor=(1.12, 1), loc='upper right')
    plt.show() 

# plot_mean_and_se_R_Time(combined_df)
plot_mean_and_se_First_Reward(combined_df)