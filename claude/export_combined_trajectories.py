#!/usr/bin/env python3
"""
Combined Trajectory Exporter
Combines all trajectory data into a single CSV file like the original format
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add claude folder to path to import the processor
sys.path.append(str(Path(__file__).parent))
from single_block_processor import CheeseboardBlock

def export_combined_trajectories(batch_output_dir: str, output_file: str = None):
    """
    Export all trajectory data from multiple experiments into a single CSV
    Similar to your original concat_trials.py format
    
    Args:
        batch_output_dir: Directory containing all processed experiments
        output_file: Output CSV file path (optional)
    """
    
    batch_dir = Path(batch_output_dir)
    if not batch_dir.exists():
        print(f"❌ Batch directory not found: {batch_dir}")
        return None
    
    if output_file is None:
        output_file = batch_dir / "combined_all_trajectories.csv"
    
    print(f"🎯 Combining trajectories from: {batch_dir}")
    print(f"📄 Output file: {output_file}")
    
    # Find all experiment directories
    exp_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith('ExperimentVideo_')]
    exp_dirs = sorted(exp_dirs)
    
    print(f"🔍 Found {len(exp_dirs)} experiment directories")
    
    combined_trajectories = []
    
    for exp_dir in exp_dirs:
        block_id = exp_dir.name
        
        # Extract date and time for grouping
        parts = block_id.split('_')
        date = parts[1] if len(parts) > 1 else 'unknown'
        time = parts[2] if len(parts) > 2 else 'unknown'
        
        print(f"📊 Processing {block_id}...")
        
        # Load trial metrics for this experiment
        metrics_file = exp_dir / f"{block_id}_trial_metrics.csv"
        if not metrics_file.exists():
            print(f"   ⚠️  No metrics file found, skipping")
            continue
        
        metrics_df = pd.read_csv(metrics_file)
        
        # Find trajectory files
        trajectory_files = list(exp_dir.glob("*_trajectory.csv"))
        
        if not trajectory_files:
            print(f"   ⚠️  No trajectory files found, skipping")
            continue
        
        # Process each bodypart
        for traj_file in trajectory_files:
            # Extract bodypart name
            bodypart = traj_file.stem.split('_')[-2]  # e.g., "nose" from "ExperimentVideo_2025-08-15_1110_nose_trajectory.csv"
            
            # Load trajectory data
            try:
                traj_df = pd.read_csv(traj_file)
                
                # Add experiment metadata
                traj_df['experiment_id'] = block_id
                traj_df['experiment_date'] = date
                traj_df['experiment_time'] = time
                traj_df['bodypart'] = bodypart
                
                # Add trial metrics for each trial
                for _, trial_row in metrics_df.iterrows():
                    trial_num = trial_row['trial_number']
                    trial_mask = traj_df['trial_number'] == trial_num
                    
                    # Add key metrics to trajectory data
                    for metric_col in ['first_reward_time', 'trial_length', 'num_rewards_collected']:
                        if metric_col in trial_row:
                            traj_df.loc[trial_mask, metric_col] = trial_row[metric_col]
                
                combined_trajectories.append(traj_df)
                
            except Exception as e:
                print(f"   ❌ Error processing {traj_file.name}: {e}")
                continue
        
        print(f"   ✅ Added {len(trajectory_files)} bodyparts")
    
    if not combined_trajectories:
        print("❌ No trajectory data found!")
        return None
    
    # Combine all trajectories
    print(f"\n🔗 Combining all trajectory data...")
    final_df = pd.concat(combined_trajectories, ignore_index=True)
    
    # Reorder columns for better readability
    column_order = [
        'experiment_id', 'experiment_date', 'experiment_time',
        'trial_number', 'frame_idx', 'frame_in_trial',
        'bodypart', 'x', 'y', 'likelihood',
        'first_reward_time', 'trial_length', 'num_rewards_collected'
    ]
    
    # Only include columns that exist
    available_cols = [col for col in column_order if col in final_df.columns]
    remaining_cols = [col for col in final_df.columns if col not in available_cols]
    final_column_order = available_cols + remaining_cols
    
    final_df = final_df[final_column_order]
    
    # Save combined file
    final_df.to_csv(output_file, index=False)
    
    print(f"\n✅ Combined trajectory export complete!")
    print(f"📊 Total records: {len(final_df):,}")
    print(f"🧪 Experiments: {final_df['experiment_id'].nunique()}")
    print(f"🎯 Bodyparts: {final_df['bodypart'].nunique()}")
    print(f"📈 Trials: {final_df['trial_number'].nunique()}")
    print(f"💾 Saved: {output_file}")
    
    return final_df

if __name__ == "__main__":
    # Default paths
    batch_dir = "/Users/nick/Projects/cheeseboardAnalysis/DATA/BATCH_ANALYSIS_ALL_DAYS"
    output_file = "/Users/nick/Projects/cheeseboardAnalysis/DATA/PROCESSED/combined_all_trajectories_complete.csv"
    
    print("🧀 Combined Trajectory Exporter")
    print("="*50)
    
    # Create output directory if needed
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Export combined trajectories
    combined_df = export_combined_trajectories(batch_dir, output_file)
    
    if combined_df is not None:
        print(f"\n📋 Preview of combined data:")
        print(combined_df.head(10))
        
        print(f"\n📊 Summary by experiment:")
        summary = combined_df.groupby(['experiment_id', 'bodypart']).size().reset_index(name='data_points')
        experiment_summary = summary.groupby('experiment_id')['data_points'].sum().reset_index()
        experiment_summary['bodyparts'] = summary.groupby('experiment_id')['bodypart'].nunique().values
        print(experiment_summary)
    else:
        print("❌ Export failed!")
