# Given a folder of csv files with trial data,
# this script will concatenate the data into a single csv file.
# It will also plot the mean and standard error per block

import pandas as pd
import os
import matplotlib.pyplot as plt

from process_timestamp import compute_trial_data

split_by_trial_folder = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/September2ndMeeting/PREPROCESSED"
output_csv = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/September2ndMeeting/combined_trials_edited.csv"

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