from experimentStruct import ExperimentStruct
import matplotlib.pyplot as plt

timestamp = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/SEPARATE/PreInterference1HourWhite/ExperimentVideo_2025-11-10_0705_timestamps.csv'
trial_data = ExperimentStruct(timestamp)

print(f"Loaded experiment with {len(trial_data.trials)} trials")
print(f"Experiment ID: {trial_data.experimentTag}")
print(f"SLEAP file: {trial_data.sleap_file}")

# Save the trial data as a pandas DataFrame that is easy to load
save_folder = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/CLEANED/'
trial_data.save_trial_data(save_folder)


# Create all plots
# trial_data.plot_durations()
# trial_data.plot_pathways()
trial_data.plot_velocity()
# trial_data.plot_occupancy_map()
# trial_data.plot_distance_traveled()
# trial_data.plot_head_direction()
# trial_data.plot_head_direction_velocity()
# trial_data.plot_angular_velocity_var()

plt.show(block=True)