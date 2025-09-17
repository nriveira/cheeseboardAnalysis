
import pandas as pd
import matplotlib.pyplot as plt

output_csv = r"C:\DATA\NICK Cheeseboard\Experiment Data\combined_experiment_data_Phase2.csv"
plot_group = [2]

combined_df = pd.read_csv(output_csv)
# Use the combined DataFrame to plot mean and standard error per block
# Use the GROUP column to plot the different groups we are comparing
def plot_mean_and_se_R_Time(combined_df):
    # Group by block and compute mean and standard error per trial
    # Use the color column to make each point a different color
    plt.figure(figsize=(10, 6))

    for i, group in enumerate(plot_group):
        group_data = combined_df[combined_df['GROUP'] == group]
        group_data['R_Time'] = group_data['R_Time'].astype('timedelta64[s]').dt.total_seconds()
        summary = group_data.groupby('Trial').agg(
            R_Time_mean=('R_Time', 'mean'),
            R_Time_sem=('R_Time', 'sem')
        ).reset_index()

        plt.errorbar(summary['Trial'] + i*0.25 - 0.25, summary['R_Time_mean'], yerr=summary['R_Time_sem'], fmt='o', capsize=5)
        # Also plot individual points for all trials
        plt.scatter(group_data['Trial'], group_data['R_Time'], alpha=0.3)

    # Plot mean and standard error, and individual points for all trials
    # Change y axis limits to 0 to 45
    plt.ylim(0, 45)

    plt.title(f'Test Reward Time Mean and SE')
    # plt.legend(['Train', 'Recall', 'Test'])  
    plt.xlabel('Trial')
    plt.ylabel('R Time (seconds)')
    plt.xticks(summary['Trial'])
    plt.show()

    # Also plot individual points for all trials
    # plt.figure(figsize=(10, 6))
    # for block in combined_df['Block'].unique():
    #     block_data = combined_df[combined_df['Block'] == block]
    #     plt.scatter(block_data['Trial'], block_data['R_Time'].astype('timedelta64[s]').dt.total_seconds(), label=f'Block {block}', alpha=0.5)
    # plt.title('Individual R Times per Trial')
    # plt.xlabel('Trial')
    # plt.ylabel('R Time (seconds)')
    # # Make the legend outside the plot
    # plt.legend(title='Block', bbox_to_anchor=(1.12, 1), loc='upper right')
    # plt.show()

def plot_mean_and_se_First_Reward(combined_df):
    # Group by block and compute mean and standard error per trial
    plt.figure(figsize=(10, 6))
    combined_df['First Reward'] = combined_df['First Reward'].astype('timedelta64[s]').dt.total_seconds()

    for i, group in enumerate(plot_group):
        group_data = combined_df[combined_df['GROUP'] == group]
        summary = group_data.groupby('Trial').agg(
            First_Reward_mean=('First Reward', 'mean'),
            First_Reward_sem=('First Reward', 'sem')
        ).reset_index()

        plt.errorbar(summary['Trial'] + i*0.25 - 0.125, summary['First_Reward_mean'], yerr=summary['First_Reward_sem'], fmt='o', capsize=5)

    plt.title('Test First Reward Mean and SE')
    # plt.legend(['Train', 'Recall', 'Test'])
    plt.xlabel('Trial')
    plt.ylabel('First Reward Time (seconds)')
    plt.xticks(summary['Trial'])
    plt.show()


plot_mean_and_se_R_Time(combined_df)
# plot_mean_and_se_First_Reward(combined_df)