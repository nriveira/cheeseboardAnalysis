# Cheeseboard Analysis Pipeline

A comprehensive, modular pipeline for processing and analyzing cheeseboard experiment data. This system integrates timestamp information with pose estimation data to provide detailed behavioral analysis and statistical modeling capabilities.

## Overview

This pipeline combines data from two sources:
1. **Timestamp data**: Frame-by-frame timing information with event markers
2. **Pose estimation data**: Coordinates and confidence scores for tracked body parts (e.g., from DeepLabCut)

The system provides a complete workflow from raw data processing to publication-ready visualizations and statistical analysis preparation.

## Features

### Core Functionality
- **Data Integration**: Seamlessly combine timestamp and pose estimation data
- **Trial Segmentation**: Automatically segment experiments into individual trials based on event markers
- **Behavioral Metrics**: Compute comprehensive trial-level metrics (timing, performance, movement)
- **Pose Trajectories**: Extract and analyze movement paths for multiple body parts
- **Modular Design**: Flexible system that adapts to different experimental designs

### Analysis Capabilities
- **Statistical Modeling Preparation**: Organize data for group comparisons and longitudinal analysis
- **Multi-session Analysis**: Combine data across experimental sessions and subjects
- **Publication-ready Visualizations**: Generate professional plots and figures
- **Comprehensive Reporting**: Automated summary statistics and metadata tracking

### Data Organization
- **Flexible Grouping**: Support for experimental groups, treatment conditions, and custom categorizations
- **Metadata Tracking**: Automatic recording of experimental parameters and processing details
- **Reproducible Workflows**: Configuration-based processing for consistent results

## Installation and Setup

### Prerequisites
```bash
pip install pandas numpy matplotlib pathlib configparser
```

### Quick Start
1. Clone or download this repository
2. Update the configuration file with your data paths
3. Run the example script to process your first experiment

## File Structure

```
cheeseboardAnalysis/
├── data_pipeline.py          # Main processing pipeline
├── visualization.py          # Plotting and visualization functions
├── example_usage.py          # Example scripts and tutorials
├── migrate_existing.py       # Migration utilities for existing data
├── experiment_config.ini     # Configuration template
├── README.md                 # This documentation
└── [legacy files]           # Your existing analysis scripts
    ├── combinedDLC.py
    ├── concat_trials.py
    ├── plotTrials.py
    └── process_timestamp.py
```

## Usage

### 1. Basic Single Experiment Processing

```python
from data_pipeline import CheeseboardDataProcessor, ExperimentConfig, DataPaths

# Configure experiment
config = ExperimentConfig(
    experiment_id="ExperimentVideo_2025-08-26_1200",
    subject_id="Mouse001",
    session_date="2025-08-26",
    experimental_group="Control",
    treatment_condition="Baseline",
    researcher="Your_Name"
)

# Set file paths
paths = DataPaths(
    timestamps_file="path/to/timestamps.csv",
    pose_estimation_file="path/to/dlc_output.csv",
    output_directory="path/to/output/"
)

# Process experiment
processor = CheeseboardDataProcessor(config, paths)
processor.load_and_integrate_data()
processor.segment_trials()
processor.compute_trial_metrics()
processor.extract_pose_trajectories(['nose', 'backOfHead'])
summary = processor.generate_summary_report()
```

### 2. Multi-Experiment Analysis

```python
from data_pipeline import DatasetManager

# Create dataset manager
manager = DatasetManager("path/to/combined_analysis/")

# Register experiments
manager.register_experiment("exp1", "path/to/exp1/output/")
manager.register_experiment("exp2", "path/to/exp2/output/")

# Combine and analyze
combined_data = manager.load_all_experiments()
analysis_data = manager.prepare_for_statistical_analysis(
    grouping_variables=['experimental_group', 'treatment_condition']
)
```

### 3. Creating Visualizations

```python
from visualization import CheeseboardVisualizer

viz = CheeseboardVisualizer("path/to/figures/")

# Plot learning curves
viz.plot_learning_curve_summary(combined_data, 'first_reward_time', 'experimental_group')

# Compare groups
viz.plot_group_comparison(combined_data, 'trial_duration', 'treatment_condition')

# Create comprehensive dashboard
viz.create_summary_dashboard(combined_data, 'first_reward_time', 'experimental_group')
```

### 4. Configuration-Based Processing

Create a configuration file (`experiment_config.ini`):

```ini
[experiment]
experiment_id = "ExperimentVideo_2025-08-26_1200"
subject_id = "Mouse001"
experimental_group = "Control"
treatment_condition = "Baseline"

[paths]
timestamps_file = "C:\\DATA\\Timestamps\\experiment_timestamps.csv"
pose_estimation_file = "C:\\DATA\\DLC\\experiment_dlc.csv"
output_directory = "C:\\DATA\\Processed\\"
```

Then process using the configuration:

```python
from example_usage import process_single_experiment

processor = process_single_experiment("experiment_config.ini")
```

## Data Formats

### Input Data

#### Timestamps File Format
Expected CSV with columns:
- `UnixTime`: Unix timestamp in nanoseconds
- `Monotonic`: Monotonic time in nanoseconds
- `Event`: Event code (0=no event, 1=start, 2=startbox, 3-5=rewards, 6=end)

#### Pose Estimation File Format
Standard DeepLabCut output format:
- Columns for each body part: `bodypart`, `bodypart.1`, `bodypart.2` (x, y, likelihood)
- Multiple rows representing video frames

### Output Data

#### Trial Metrics
- `trial_duration`: Total time from start to end
- `first_reward_time`: Time to first reward collection
- `reward_collection_duration`: Time between first and last reward
- `startbox_to_first_reward`: Time from startbox exit to first reward
- Plus experimental metadata (subject, group, condition, etc.)

#### Pose Trajectories
- Frame-by-frame coordinates for each body part
- Linked to trial numbers and experimental metadata
- Confidence filtering available

## Event Coding System

Default event codes (customizable):
- `1`: Trial start
- `2`: Startbox entry/exit
- `3`: Reward 1 collection
- `4`: Reward 2 collection  
- `5`: Reward 3 collection
- `6`: Trial end

## Migration from Existing Code

If you have existing analysis code, use the migration utilities:

```python
from migrate_existing import migrate_existing_experiment, batch_process_existing_experiments

# Migrate single experiment
processor = migrate_existing_experiment(
    timestamps_file="existing_timestamps.csv",
    dlc_file="existing_dlc.csv", 
    output_dir="new_output/",
    experiment_info={"experiment_id": "exp1", "subject_id": "mouse1"}
)

# Batch process multiple experiments
batch_process_existing_experiments()
```

## Statistical Analysis Preparation

The pipeline prepares data for statistical analysis by:

1. **Standardizing Variables**: Consistent naming and formatting across experiments
2. **Adding Derived Variables**: Session order, trial within session, performance indices
3. **Grouping Support**: Flexible categorization for factorial designs
4. **Missing Data Handling**: Appropriate treatment of incomplete trials
5. **Export Formats**: CSV files ready for R, Python statsmodels, or other statistical software

### Example Statistical Models

The processed data supports various analyses:

```python
# Example using statsmodels (Python)
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm

# Mixed-effects model for learning over trials
model = mixedlm("first_reward_time ~ trial_number * experimental_group", 
                analysis_data, groups=analysis_data["subject_id"])
result = model.fit()
```

## Customization

### Adding Custom Metrics
Extend the `compute_trial_metrics()` method to include experiment-specific measures:

```python
def compute_custom_metrics(self, trial_data):
    # Add your custom calculations
    custom_metrics = {}
    # ... your code here
    return custom_metrics
```

### Custom Visualizations
Create experiment-specific plots by extending the `CheeseboardVisualizer` class:

```python
class CustomVisualizer(CheeseboardVisualizer):
    def plot_experiment_specific(self, data):
        # Your custom plotting code
        pass
```

## Performance and Scalability

- **Memory Efficient**: Processes large datasets in chunks
- **Parallel Processing**: Support for batch processing multiple experiments
- **Caching**: Intermediate results saved to avoid recomputation
- **Progress Tracking**: Detailed logging for long-running analyses

## Troubleshooting

### Common Issues

1. **File Path Errors**: Ensure all paths use absolute paths and correct separators for your OS
2. **Data Format Mismatches**: Check that your timestamp and pose files have expected columns
3. **Memory Issues**: For very large experiments, consider processing trials individually
4. **Missing Events**: Verify event coding matches your experimental protocol

### Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Getting Help

1. Check the example usage scripts for common patterns
2. Review the migration utilities if adapting existing code
3. Examine the docstrings for detailed parameter descriptions

## Contributing

To contribute to this pipeline:
1. Follow the existing code structure and documentation style
2. Add unit tests for new functionality
3. Update this README with any new features
4. Ensure backward compatibility with existing data formats

## License

This project is part of ongoing neuroscience research. Please cite appropriately if used in publications.

## Changelog

### Version 1.0 (August 2025)
- Initial modular pipeline implementation
- Comprehensive data integration and analysis tools
- Publication-ready visualization suite
- Migration utilities for existing code
- Statistical analysis preparation tools

