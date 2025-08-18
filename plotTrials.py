
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

processed_csv = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/ExperimentVideo_2025-08-15_1110_preprocessed.csv"
split_by_trial_csv = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/ExperimentVideo_2025-08-15_1110_split_by_trial.csv"

def plot_trial_path(preprocessed, trial_split, trial_number=0, bodypart='nose'):
    preprocessed_df = pd.read_csv(preprocessed)
    trial_split_df = pd.read_csv(trial_split)

    trial_data = trial_split_df.iloc[trial_number]
    start_time = trial_data['SB_idx']
    end_time = trial_data['End_idx']

    trial_df = preprocessed_df.iloc[start_time:end_time + 1]

    coords = np.empty((3, end_time - start_time + 1))
    col_index = 0
    array_index = 0
    # Get the bodypart coordinates
    for col in preprocessed_df.iloc[0,:]:
        if col == bodypart:
            coords[array_index, :] = trial_df.iloc[:, col_index].values
            array_index += 1
        col_index += 1

    plt.figure(figsize=(10, 6))
    x_coord = coords[0]
    y_coord = coords[1]

    # If the 3rd coordinate is < 0.5, set coords 0, 1 to NaN
    if coords.shape[0] > 2:
        x_coord[coords[2] < 0.5] = np.nan
        y_coord[coords[2] < 0.5] = np.nan
    
    plt.plot(x_coord, y_coord)
    plt.title(f'Trial {trial_number + 1} Path for {bodypart} (Filtered p<0.5)')
    plt.show()

plot_trial_path(processed_csv, split_by_trial_csv, trial_number=6, bodypart='nose')