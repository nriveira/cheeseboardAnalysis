
import pandas as pd
import matplotlib.pyplot as plt

output_csv = r"C:\DATA\NICK Cheeseboard\Experiment Data\combined_experiment_data_Phase2.csv"

combined_df = pd.read_csv(output_csv)
# Use the combined DataFrame to plot mean and standard error per block
# Use the GROUP column to plot the different groups we are comparing
def plot_mean_and_se_R_Time(combined_df):
    # Group by block and compute mean and standard error per trial
    plt.figure(figsize=(10, 6))

    for i, group in enumerate([0,1,2]):
        group_data = combined_df[combined_df['GROUP'] == group]
        group_data['R_Time'] = group_data['R_Time'].astype('timedelta64[s]').dt.total_seconds()
        summary = group_data.groupby('Trial').agg(
            R_Time_mean=('R_Time', 'mean'),
            R_Time_sem=('R_Time', 'sem')
        ).reset_index()

        plt.errorbar(summary['Trial'] + i*0.25 - 0.25, summary['R_Time_mean'], yerr=summary['R_Time_sem'], fmt='o', capsize=5)

    # Plot mean and standard error, and individual points for all trials
    plt.title(f'Test Reward Time Mean and SE Break vs No Break')
    plt.legend(['Train', 'Recall', 'Test'])  
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

    for i, group in enumerate([0,1,2]):
        group_data = combined_df[combined_df['GROUP'] == group]
        summary = group_data.groupby('Trial').agg(
            First_Reward_mean=('First Reward', 'mean'),
            First_Reward_sem=('First Reward', 'sem')
        ).reset_index()

        plt.errorbar(summary['Trial'] + i*0.25 - 0.125, summary['First_Reward_mean'], yerr=summary['First_Reward_sem'], fmt='o', capsize=5)

    plt.title('Test First Reward Mean and SE Break vs No Break')
    plt.legend(['Train', 'Recall', 'Test'])
    plt.xlabel('Trial')
    plt.ylabel('First Reward Time (seconds)')
    plt.xticks(summary['Trial'])
    plt.show()

# Plot the performance of the first trial for all time points in a group
def plot_group_over_time(combined_df):
    plt.figure(figsize=(10, 6))
    for group in [0]:
        group_data = combined_df[combined_df['GROUP'] == group]
        # Pull only the first three trials from each datetime
        group_data['First Reward'] = group_data['First Reward'].astype('timedelta64[s]').dt.total_seconds()
        group_data['R_Time'] = group_data['R_Time'].astype('timedelta64[s]').dt.total_seconds()

        group_data = group_data[group_data['Trial'].isin([0, 1, 2])]
        # Get the average of the three trials
        group_data = group_data.groupby('Datetime').agg(
            First_Reward=('First Reward', 'mean'),
            R_Time=('R_Time', 'mean')
        ).reset_index()

        plt.bar(group_data['Datetime'], group_data['First_Reward'], label=f'Group {group}')
        #plt.plot(group_data['Datetime'], group_data['R_Time'], label=f'Group {group} R Time')


    # Reduce the number of listed times to make the graph more readable
    plt.xticks(rotation=45)
    plt.title('First Reward Over Time by Group')
    plt.xlabel('Datetime')
    plt.ylabel('First Reward Time (seconds)')
    plt.legend()
    plt.show()

plot_mean_and_se_R_Time(combined_df)
# plot_mean_and_se_First_Reward(combined_df)
# plot_group_over_time(combined_df)