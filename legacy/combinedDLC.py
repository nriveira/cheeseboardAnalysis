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
    dlc_df = pd.read_csv(dlc_csv, header=None)

    # Combine the second and third rows of the DLC file for each column and make that the first row
    bodypart_header = dlc_df.iloc[1].values
    coord_header = dlc_df.iloc[2].values

    # Concatenate the strings of all headers into one
    dlc_header = pd.Series([f"{bodypart}_{coord}" for bodypart, coord in zip(bodypart_header, coord_header)])

    # Make dlc_header the header of the DLC DataFrame
    dlc_df.columns = dlc_header

    # Remove the original headers
    dlc_df = dlc_df[3:].reset_index(drop=True)

    # Combine the DataFrames
    combined_df = pd.concat([timestamps_df, dlc_df], axis=1)
    
    # Write the combined DataFrame to a new CSV file
    combined_df.to_csv(output_csv, index=False)

combine_timestamps_with_dlc(time_file, dlc_file, output_file)