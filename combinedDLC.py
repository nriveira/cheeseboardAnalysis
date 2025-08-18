# Combine timestamp data with DLC data
# Both CSV files will contain the same number of rows
# Simply concatenate the two DataFrames
# DLC data contains 3 header columns, so we need to handle that

dlc_file = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA/ExperimentVideo_2025-08-13_1105DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
time_file = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA/ExperimentVideo_2025-08-13_1105_timestamps.csv"
output_file = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/PREPROCESSED/ExperimentVideo_2025-08-13_1105_preprocessed.csv"

import pandas as pd

def combine_timestamps_with_dlc(timestamps_csv, dlc_csv, output_csv):
    # Read the CSV files
    timestamps_df = pd.read_csv(timestamps_csv, header=None, names=['UnixTime', 'Monotonic', 'Event'])
    dlc_df = pd.read_csv(dlc_csv)

    # Create a new DataFrame with 3 rows of 0s, and the columns from the timestamps DataFrame
    timestamps_header = pd.DataFrame(0, index=range(2), columns=['UnixTime', 'Monotonic', 'Event'])

    # Add the timestamps to the header DataFrame
    timestamps_header = pd.concat([timestamps_header, timestamps_df], axis=0, ignore_index=True)

    # # Combine the DataFrames
    combined_df = pd.concat([timestamps_header, dlc_df], axis=1)
    
    # # Write the combined DataFrame to a new CSV file
    combined_df.to_csv(output_csv, index=False)

combine_timestamps_with_dlc(time_file, dlc_file, output_file)