import sleapAnalysis


timestamp = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/ExperimentVideo_2025-11-10_0738_timestamps.csv'

trial_data = sleapAnalysis.sleapAndTimestamp(timestamp)

trial_data.plot_durations()
trial_data.plot_pathways(tracking_part='neck1')
trial_data.plot_velocity()