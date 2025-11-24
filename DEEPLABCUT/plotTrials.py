
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sleap_csv = r"C:\DATA\NICK Cheeseboard\SLEAP Tracking\labels.v001.077_ExperimentVideo_2025-09-04_1013.analysis.csv"
split_by_trial_csv = r"C:\DATA\NICK Cheeseboard\Experiment Trials\ExperimentVideo_2025-09-04_1013_split_by_trial.csv"
save_dir = r"C:\DATA\NICK Cheeseboard\Experiment Data"
desc = "2025-09-04_1013"

def get_trial_path(sleap, trial_split, trial_number=1, bodypart='backOfHead'):
    sleap_df = pd.read_csv(sleap)
    trial_split_df = pd.read_csv(trial_split)

    trial_data = trial_split_df.iloc[trial_number]
    start_time = trial_data['Start_idx']
    end_time = trial_data['End_idx']

    if pd.isna(start_time) or pd.isna(end_time):
        return None, None
    else:
        start_time = int(start_time)
        end_time = int(end_time)

    # Find the start and end times using the sleap_df column 'frame_idx'
    trial_df = sleap_df[(sleap_df['frame_idx'] >= start_time) & (sleap_df['frame_idx'] <= end_time)].reset_index(drop=True)

    # Only keep columns that have the bodypart name in them
    bodypart_columns = [col for col in trial_df.columns if bodypart in col]
    trial_df = trial_df[bodypart_columns]
    return trial_df

# Plot the path of each trial
def plot_trial_paths(sleap, trial_split, bodypart='nose'):
    trial_split_df = pd.read_csv(trial_split)

    # Make one panel to plot all trials, with subplots in groups of 5.
    # If one of the plots is blank, remove the corresponding axis.
    num_trials = len(trial_split_df)
    num_cols = 5
    num_rows = (num_trials + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 3 * num_rows))
    axs = axs.flatten()

    axis_i = 0
    for i in range(num_trials):
        bp = get_trial_path(sleap, trial_split, i, bodypart)
        if bp is not None:
            # The headers are in the format bodypart.x, bodypart.y, bodypart.score
            bodypart_x = bp.iloc[:, 0]
            bodypart_y = bp.iloc[:, 1]
            axs[axis_i].plot(bodypart_x, bodypart_y, label=f'Trial {i + 1}')
            axs[axis_i].set_title(f'Trial {i + 1}')

            axis_i += 1

    plt.suptitle(f'Path for all trials - {desc}')
    plt.savefig(f"{save_dir}Block{desc}_{bodypart}.png")

# Plot the path of the backOfHead for each trial
plot_trial_paths(sleap_csv, split_by_trial_csv, bodypart='nose')
# plot_trial_paths(sleap_csv, split_by_trial_csv, bodypart='backOfHead')
# plot_trial_paths(sleap_csv, split_by_trial_csv, bodypart='earR')
# plot_trial_paths(sleap_csv, split_by_trial_csv, bodypart='earL')
# plot_trial_paths(sleap_csv, split_by_trial_csv, bodypart='baseOfTail')
# plot_trial_paths(sleap_csv, split_by_trial_csv, bodypart='pawFL')