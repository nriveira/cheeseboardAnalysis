
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

processed_csv = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/PREPROCESSED/ExperimentVideo_2025-08-15_1207_preprocessed.csv"
split_by_trial_csv = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/PREPROCESSED/ExperimentVideo_2025-08-15_1207_split_by_trial.csv"
save_dir = r"/Users/nick/Projects/cheeseboardAnalysis/DATA/Figures/"
desc = "2025-08-15_1207"

def get_trial_path(preprocessed, trial_split, trial_number=1, bodypart='backOfHead'):
    preprocessed_df = pd.read_csv(preprocessed)
    trial_split_df = pd.read_csv(trial_split)

    trial_data = trial_split_df.iloc[trial_number]
    start_time = trial_data['Start_idx']
    end_time = trial_data['End_idx']

    if pd.isna(start_time) or pd.isna(end_time):
        return None, None
    else:
        start_time = int(start_time)
        end_time = int(end_time)

    trial_df = preprocessed_df.iloc[start_time:end_time + 1]

    coords = np.empty((3, end_time - start_time + 1))

    # Get the bodypart coordinates by finding the headers from the bodypart
    bodypart_headers = [col for col in trial_df.columns if bodypart in col]
    if len(bodypart_headers) < 3:
        return None, None

    coords[0] = trial_df[bodypart_headers[0]].values
    coords[1] = trial_df[bodypart_headers[1]].values
    coords[2] = trial_df[bodypart_headers[2]].values

    # Print the percentage of coordinates with likelihood above 0.5
    if coords.shape[0] > 2:
        valid_coords = coords[2] >= 0.5
        print(f"Percentage of valid coordinates for trial {trial_number}: {np.sum(valid_coords) / valid_coords.size * 100:.2f}%")

    # If the 3rd coordinate is < 0.5, set coords 0, 1 to NaN
    if coords.shape[0] > 2:
        coords[0][coords[2] < 0.5] = np.nan
        coords[1][coords[2] < 0.5] = np.nan

    return coords[0], coords[1]

# Plot the path of each trial
def plot_trial_paths(preprocessed, trial_split, bodypart='nose'):
    trial_split_df = pd.read_csv(trial_split)

    # Make one panel to plot all trials, with subplots in groups of 5.
    # If one of the plots is blank, remove the corresponding axis.
    num_trials = len(trial_split_df)
    num_cols = 5
    num_rows = 2#(num_trials + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axs = axs.flatten()

    axis_i = 0
    for i in range(num_trials):
        x, y = get_trial_path(preprocessed, trial_split, i, bodypart)
        if x is not None and y is not None:
            axs[axis_i].plot(x, y, label=f'Trial {i + 1}')

            # Indicate the start and end positions
            axs[axis_i].scatter([x[0]], [y[0]], color='green', label='Start')
            axs[axis_i].scatter([x[-1]], [y[-1]], color='red', label='End')
            axs[axis_i].set_title(f'Trial {i + 1}')

            axis_i += 1

    plt.suptitle(f'Path for all trials - {desc}')
    plt.savefig(f"{save_dir}Block{desc}_{bodypart}.png")

# Plot the path of the backOfHead for each trial
plot_trial_paths(processed_csv, split_by_trial_csv, bodypart='nose')
