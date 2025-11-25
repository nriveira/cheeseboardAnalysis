from plotDurations import DurationPlot
from plotPathway import PathwayPlot

timestamp = r'/Users/nick/Projects/cheeseboardAnalysis/DATA/NOVEMBER/ExperimentVideo_2025-11-10_0705_timestamps.csv'

DurationPlot.plot_durations(timestamp)
PathwayPlot.plot_pathways(timestamp, tracking_part='nose1')