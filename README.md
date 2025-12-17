# Cheeseboard Analysis

Code for analyzing cheeseboard data. Code is split by pose estimation software used, DEEPLABCUT was first iteration and depreciated, we now use SLEAP.

## DEEPLABCUT code

TODO: Update descriptions of DeepLabCut code, written by both me and Copilot (Claude)

## SLEAP

### Current Goals

* Visualize single trial data for SLEAP verification
* Create preliminary behavioral analysis based on distance traveled and speed
* Make putative headscan detection using SLEAP tracking

### Future Goals

* Implement SIMBA for further behavioral analysis
* Improve behavioral analysis methods, with things like optimal and linearized path.
* Automate generation of spatial locations, based on more descriptive criteria required for task (Side preference, differentialting spaces between trials, etc.)
* Better editing of timestamp files if mistakes made during recording

### dashboard.py

Space to run all code for the pipeline

### sleapAnalysis

Converts input cherubim structure to analyzed structure for individual trial indexing.

Functions:

* find_durations
* find_velocity(tracking_part=neck1)
* plot_durations
* plot_pathways(tracking_part=neck1)
* plot_velocity

### experimentStruct

Replacement for cheeseboardTimestamp file, directly structuring sessions from file to index individual trials

### analysisStruct.py
