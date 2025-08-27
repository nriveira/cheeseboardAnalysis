"""
Single Block Processor for Cheeseboard Analysis
==============================================

This module focuses on processing a single block (video) of cheeseboard data.
It combines timestamp and DLC data, segments trials, computes metrics, and creates visualizations.

A "block" corresponds to one video recording session containing multiple trials.
Each trial is defined by event markers (1=start, 2=startbox, 3-5=rewards, 6=end).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple


class CheeseboardBlock:
    """
    Class for processing a single block of cheeseboard experiment data.
    
    A block contains:
    - Timestamp data with event markers
    - DLC pose estimation data
    - Multiple trials within the block
    """
    
    def __init__(self, block_id: str, timestamps_file: str, dlc_file: str, output_dir: str = None):
        """
        Initialize a CheeseboardBlock processor
        
        Args:
            block_id: Unique identifier for this block (e.g., "ExperimentVideo_2025-08-20_1043")
            timestamps_file: Path to timestamp CSV file
            dlc_file: Path to DLC pose estimation CSV file
            output_dir: Directory to save processed data (optional)
        """
        self.block_id = block_id
        self.timestamps_file = timestamps_file
        self.dlc_file = dlc_file
        
        # Set up output directory
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path.cwd() / "output" / block_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data containers
        self.raw_timestamps = None
        self.raw_dlc = None
        self.integrated_data = None
        self.trial_segments = None
        self.trial_metrics = None
        self.bodypart_data = {}
        
        print(f"🧀 Initialized CheeseboardBlock: {block_id}")
        print(f"   📁 Output directory: {self.output_dir}")
    
    def load_data(self) -> bool:
        """
        Load the raw timestamp and DLC data files
        DLC files have 3 header rows that need to be combined into meaningful column names
        
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"📂 Loading data for block {self.block_id}...")
        
        try:
            # Load timestamp data
            self.raw_timestamps = pd.read_csv(
                self.timestamps_file, 
                header=None, 
                names=['UnixTime', 'Monotonic', 'Event']
            )
            print(f"   ✅ Loaded {len(self.raw_timestamps)} timestamp rows")
            
            # Load DLC data with special handling for 3-row header
            print(f"   🔄 Processing DLC file with 3-row header...")
            self.raw_dlc = self._load_dlc_with_combined_headers()
            print(f"   ✅ Loaded {len(self.raw_dlc)} DLC rows with {len(self.raw_dlc.columns)} columns")
            
            # Basic validation
            if len(self.raw_timestamps) != len(self.raw_dlc):
                print(f"   ⚠️  Row count mismatch: timestamps={len(self.raw_timestamps)}, dlc={len(self.raw_dlc)}")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error loading data: {e}")
            return False
    
    def _load_dlc_with_combined_headers(self) -> pd.DataFrame:
        """
        Load DLC file and combine the 3 header rows into meaningful column names
        
        DLC structure:
        Row 0: model info (e.g., 'DLC_resnet50_liveTestAug8shuffle1_100000')
        Row 1: bodypart names (e.g., 'nose', 'backOfHead')  
        Row 2: coordinate type (e.g., 'x', 'y', 'likelihood')
        
        Combined result: 'nose_x', 'nose_y', 'nose_likelihood', etc.
        
        Returns:
            pd.DataFrame: DLC data with combined headers
        """
        # Read the DLC file with all headers as data initially
        raw_df = pd.read_csv(self.dlc_file, header=None)
        
        if len(raw_df) < 4:  # Need at least 3 header rows + 1 data row
            raise ValueError("DLC file appears to be too short or missing header rows")
        
        # Extract the 3 header rows
        model_row = raw_df.iloc[0].fillna('')  # Row 0: model info
        bodypart_row = raw_df.iloc[1].fillna('')  # Row 1: bodypart names
        coord_row = raw_df.iloc[2].fillna('')  # Row 2: x/y/likelihood
        
        # Create combined column names
        combined_columns = []
        for i in range(len(bodypart_row)):
            model = str(model_row.iloc[i]).strip()
            bodypart = str(bodypart_row.iloc[i]).strip()
            coord = str(coord_row.iloc[i]).strip()
            
            # Skip empty columns
            if not bodypart or bodypart == '' or bodypart.lower() == 'nan':
                combined_columns.append(f"empty_col_{i}")
                continue
            
            # Create meaningful column name
            if coord.lower() in ['x', 'y', 'likelihood']:
                combined_name = f"{bodypart}_{coord}"
            else:
                combined_name = f"{bodypart}_{coord}" if coord else bodypart
            
            combined_columns.append(combined_name)
        
        # Get the actual data (skip the 3 header rows)
        data_df = raw_df.iloc[3:].reset_index(drop=True)
        data_df.columns = combined_columns
        
        # Convert numeric columns
        for col in data_df.columns:
            if '_x' in col or '_y' in col or '_likelihood' in col:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
        
        print(f"     🔄 Combined headers: {len([c for c in combined_columns if 'empty_col' not in c])} meaningful columns")
        print(f"     📝 Example columns: {combined_columns[:5]}")
        
        return data_df
    
    def integrate_data(self) -> pd.DataFrame:
        """
        Combine timestamp and DLC data into a single DataFrame
        (Based on your combinedDLC.py approach)
        
        Returns:
            pd.DataFrame: Integrated dataset
        """
        print(f"🔗 Integrating timestamp and DLC data...")
        
        if self.raw_timestamps is None or self.raw_dlc is None:
            print("   ❌ Must load data first")
            return None
        
        # Create header rows for timestamps (to match DLC structure)
        timestamps_header = pd.DataFrame(
            0, 
            index=range(2), 
            columns=['UnixTime', 'Monotonic', 'Event']
        )
        
        # Combine header with actual timestamp data
        timestamps_with_header = pd.concat([
            timestamps_header, 
            self.raw_timestamps
        ], axis=0, ignore_index=True)
        
        # Combine timestamps and DLC data side by side
        self.integrated_data = pd.concat([
            timestamps_with_header, 
            self.raw_dlc
        ], axis=1)
        
        # Save integrated data
        integrated_file = self.output_dir / f"{self.block_id}_integrated.csv"
        self.integrated_data.to_csv(integrated_file, index=False)
        
        print(f"   ✅ Created integrated dataset: {self.integrated_data.shape}")
        print(f"   💾 Saved: {integrated_file}")
        
        return self.integrated_data
    
    def segment_trials(self) -> pd.DataFrame:
        """
        Segment the data into individual trials based on event markers
        (Based on your process_timestamp.py approach)
        
        Returns:
            pd.DataFrame: Trial segmentation information
        """
        print(f"✂️  Segmenting trials for block {self.block_id}...")
        
        if self.integrated_data is None:
            print("   ❌ Must integrate data first")
            return None
        
        # Get only rows with events (not 0)
        event_data = self.integrated_data[self.integrated_data['Event'] != 0].copy()
        event_data['Index'] = event_data.index
        
        # Convert timestamps
        event_data['Time'] = pd.to_datetime(event_data['UnixTime'].astype('int64'), unit='ns')
        event_data['Duration'] = pd.to_timedelta(event_data['Monotonic'].astype('int64'), unit='ns')
        
        # Make duration relative to start
        if len(event_data) > 0:
            event_data['Duration'] = event_data['Duration'] - event_data['Duration'].iloc[0]
        
        event_data = event_data.reset_index(drop=True)
        
        # Create trial segments
        trials = []
        current_trial = {}
        trial_number = 0
        
        event_names = {1: 'Start', 2: 'SB', 3: 'R1', 4: 'R2', 5: 'R3', 6: 'End'}
        
        for i, row in event_data.iterrows():
            event = row['Event']
            event_name = event_names.get(event, f'Event{event}')
            
            if event == 1:  # Start of new trial
                if current_trial:  # Save previous trial
                    current_trial['trial_number'] = trial_number
                    trials.append(current_trial.copy())
                    trial_number += 1
                
                # Start new trial
                current_trial = {
                    'block_id': self.block_id,
                    f'{event_name}_idx': row['Index'],
                    f'{event_name}_Duration': row['Duration'],
                    f'{event_name}_Time': row['Time']
                }
            else:
                # Add event to current trial
                if current_trial:  # Only if we have a trial started
                    current_trial[f'{event_name}_idx'] = row['Index']
                    current_trial[f'{event_name}_Duration'] = row['Duration']
                    current_trial[f'{event_name}_Time'] = row['Time']
        
        # Don't forget the last trial
        if current_trial:
            current_trial['trial_number'] = trial_number
            trials.append(current_trial)
        
        self.trial_segments = pd.DataFrame(trials)
        
        # Save trial segments
        segments_file = self.output_dir / f"{self.block_id}_trial_segments.csv"
        self.trial_segments.to_csv(segments_file, index=False)
        
        print(f"   ✅ Found {len(self.trial_segments)} trials")
        print(f"   💾 Saved: {segments_file}")
        
        return self.trial_segments
    
    def compute_trial_metrics(self) -> pd.DataFrame:
        """
        Compute behavioral metrics for each trial
        Only requires start and end events - reward events are optional
        
        Returns:
            pd.DataFrame: Trial metrics
        """
        print(f"📊 Computing trial metrics for block {self.block_id}...")
        
        if self.trial_segments is None:
            print("   ❌ Must segment trials first")
            return None
        
        metrics = []
        
        for _, trial in self.trial_segments.iterrows():
            trial_metrics = {
                'block_id': trial['block_id'],
                'trial_number': trial['trial_number'],
            }
            
            # Get timing data - only require start, end is optional
            start_duration = trial.get('Start_Duration')
            end_duration = trial.get('End_Duration')
            
            # Trial length (only if both start and end are available)
            if pd.notna(start_duration) and pd.notna(end_duration):
                trial_metrics['trial_length'] = end_duration - start_duration
            elif pd.notna(start_duration):
                trial_metrics['trial_length'] = None  # Will be handled gracefully
            
            # Individual reward times (relative to trial start) - all optional
            reward_times = []
            for reward in ['R1', 'R2', 'R3']:
                reward_duration = trial.get(f'{reward}_Duration')
                if pd.notna(reward_duration) and pd.notna(start_duration):
                    reward_time = reward_duration - start_duration
                    trial_metrics[f'{reward.lower()}_time'] = reward_time
                    reward_times.append(reward_time)
                else:
                    trial_metrics[f'{reward.lower()}_time'] = None
            
            # Reward collection metrics (only if at least one reward was collected)
            if reward_times:
                trial_metrics['first_reward_time'] = min(reward_times)
                trial_metrics['last_reward_time'] = max(reward_times)
                trial_metrics['reward_collection_duration'] = max(reward_times) - min(reward_times)
                trial_metrics['num_rewards_collected'] = len(reward_times)
                
                # Time after last reward (only if end is available)
                if pd.notna(end_duration) and pd.notna(start_duration):
                    trial_metrics['time_after_last_reward'] = end_duration - (start_duration + max(reward_times))
            else:
                # No rewards collected
                trial_metrics['first_reward_time'] = None
                trial_metrics['last_reward_time'] = None
                trial_metrics['reward_collection_duration'] = None
                trial_metrics['num_rewards_collected'] = 0
                trial_metrics['time_after_last_reward'] = None
            
            # StartBox metrics (optional)
            sb_duration = trial.get('SB_Duration')
            if pd.notna(sb_duration) and pd.notna(start_duration):
                trial_metrics['startbox_time'] = sb_duration - start_duration
                if reward_times:
                    trial_metrics['startbox_to_first_reward'] = min(reward_times) - (sb_duration - start_duration)
                else:
                    trial_metrics['startbox_to_first_reward'] = None
            else:
                trial_metrics['startbox_time'] = None
                trial_metrics['startbox_to_first_reward'] = None
            
            metrics.append(trial_metrics)
        
        self.trial_metrics = pd.DataFrame(metrics)
        
        # Save metrics
        metrics_file = self.output_dir / f"{self.block_id}_trial_metrics.csv"
        self.trial_metrics.to_csv(metrics_file, index=False)
        
        print(f"   ✅ Computed metrics for {len(self.trial_metrics)} trials")
        print(f"   💾 Saved: {metrics_file}")
        
        return self.trial_metrics
    
    def extract_bodypart_trajectories(self, bodyparts: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Extract movement trajectories for specified body parts
        
        Args:
            bodyparts: List of body part names to extract. If None, detect automatically.
        
        Returns:
            Dict mapping bodypart names to trajectory DataFrames
        """
        print(f"🎯 Extracting trajectories for block {self.block_id}...")
        
        if self.integrated_data is None or self.trial_segments is None:
            print("   ❌ Must integrate data and segment trials first")
            return {}
        
        # Auto-detect bodyparts if not specified
        if bodyparts is None:
            bodyparts = self._detect_bodyparts()
        
        trajectories = {}
        
        for bodypart in bodyparts:
            print(f"   📍 Processing {bodypart}...")
            
            bodypart_trajectories = []
            
            for _, trial in self.trial_segments.iterrows():
                start_idx = trial.get('Start_idx')
                end_idx = trial.get('End_idx')
                
                # Skip trials that don't have both start and end markers
                if pd.isna(start_idx):
                    print(f"     ⚠️  Trial {trial.get('trial_number', '?')} missing start marker, skipping")
                    continue
                
                if pd.isna(end_idx):
                    print(f"     ⚠️  Trial {trial.get('trial_number', '?')} missing end marker, using data end")
                    end_idx = len(self.integrated_data) - 1
                
                start_idx, end_idx = int(start_idx), int(end_idx)
                trial_data = self.integrated_data.iloc[start_idx:end_idx + 1]
                
                # Get coordinate columns for this bodypart using new combined naming
                x_col = f"{bodypart}_x"
                y_col = f"{bodypart}_y"
                likelihood_col = f"{bodypart}_likelihood"
                
                if x_col in trial_data.columns and y_col in trial_data.columns:
                    for frame_idx, frame in trial_data.iterrows():
                        try:
                            x = frame[x_col] if pd.notna(frame[x_col]) and isinstance(frame[x_col], (int, float)) else np.nan
                            y = frame[y_col] if pd.notna(frame[y_col]) and isinstance(frame[y_col], (int, float)) else np.nan
                            likelihood = frame[likelihood_col] if likelihood_col in trial_data.columns and pd.notna(frame[likelihood_col]) and isinstance(frame[likelihood_col], (int, float)) else 1.0
                            
                            # Apply confidence threshold
                            if likelihood < 0.5:
                                x, y = np.nan, np.nan
                            
                            bodypart_trajectories.append({
                                'block_id': self.block_id,
                                'trial_number': trial['trial_number'],
                                'frame_idx': frame_idx,
                                'frame_in_trial': frame_idx - start_idx,
                                'bodypart': bodypart,
                                'x': x,
                                'y': y,
                                'likelihood': likelihood
                            })
                        except Exception as e:
                            # Skip problematic frames
                            continue
                else:
                    print(f"     ⚠️  Missing coordinate columns for {bodypart} (looking for {x_col}, {y_col})")
            
            if bodypart_trajectories:
                trajectories[bodypart] = pd.DataFrame(bodypart_trajectories)
                
                # Save trajectory data
                traj_file = self.output_dir / f"{self.block_id}_{bodypart}_trajectory.csv"
                trajectories[bodypart].to_csv(traj_file, index=False)
                print(f"     💾 Saved: {traj_file}")
        
        self.bodypart_data = trajectories
        return trajectories
    
    def _detect_bodyparts(self) -> List[str]:
        """
        Detect bodypart names from the combined DLC column headers
        
        Returns:
            List[str]: List of detected bodypart names
        """
        if self.combined_data is None:
            return []
        
        bodyparts = set()
        
        # Look for columns with the pattern: bodypart_coordinate
        for col in self.combined_data.columns:
            if '_x' in col or '_y' in col or '_likelihood' in col:
                # Extract bodypart name (everything before the last underscore)
                parts = col.split('_')
                if len(parts) >= 2:
                    # Get everything except the last part (which should be x, y, or likelihood)
                    bodypart = '_'.join(parts[:-1])
                    if bodypart and not bodypart.startswith('empty_col'):
                        bodyparts.add(bodypart)
        
        detected = sorted(list(bodyparts))
        print(f"     🎯 Detected bodyparts from combined headers: {detected}")
        
        return detected
    
    def inspect_dlc_structure(self) -> None:
        """
        Print the DLC file structure to help identify bodyparts manually
        """
        if self.raw_dlc is None:
            print("   ❌ No DLC data loaded")
            return
        
        print(f"\n🔍 DLC File Structure for {self.block_id}")
        print("=" * 50)
        print(f"Shape: {self.raw_dlc.shape}")
        print(f"Columns: {len(self.raw_dlc.columns)}")
        
        print("\nFirst 10 columns:")
        for i, col in enumerate(self.raw_dlc.columns[:10]):
            print(f"  {i}: {col}")
        
        print("\nFirst 3 rows:")
        print(self.raw_dlc.head(3))
        
        print("\nColumn data types:")
        print(self.raw_dlc.dtypes.head(10))
        
        # Look for potential bodypart patterns
        print("\nPotential bodypart patterns:")
        for i, col in enumerate(self.raw_dlc.columns):
            if i < 20:  # Check first 20 columns
                col_str = str(col)
                if not any(x in col_str.lower() for x in ['unnamed']):
                    print(f"  Column {i}: '{col}' -> y: '{col}.1'?, likelihood: '{col}.2'?")
        
        print("\n" + "=" * 50)
    
    def create_trial_plots(self, bodypart: str = None, save_individual: bool = True, save_combined: bool = True) -> None:
        """
        Create visualizations for trial trajectories
        
        Args:
            bodypart: Bodypart to plot. If None, uses first available.
            save_individual: Whether to save individual trial plots
            save_combined: Whether to save combined plot with all trials
        """
        print(f"📊 Creating plots for block {self.block_id}...")
        
        # Choose bodypart
        if bodypart is None:
            if self.bodypart_data:
                bodypart = list(self.bodypart_data.keys())[0]
            else:
                print("   ❌ No trajectory data available. Run extract_bodypart_trajectories first.")
                return
        
        if bodypart not in self.bodypart_data:
            print(f"   ❌ No data for bodypart '{bodypart}'. Available: {list(self.bodypart_data.keys())}")
            return
        
        trajectory_df = self.bodypart_data[bodypart]
        trials = trajectory_df['trial_number'].unique()
        
        # Save plots directly to the main output directory (not in a subdirectory)
        plots_dir = self.output_dir
        
        # Individual trial plots
        if save_individual:
            print(f"   📈 Creating {len(trials)} individual trial plots...")
            for trial_num in trials:
                trial_data = trajectory_df[trajectory_df['trial_number'] == trial_num]
                
                plt.figure(figsize=(10, 8))
                plt.plot(trial_data['x'], trial_data['y'], 'b-', linewidth=2, alpha=0.7, label='Path')
                
                # Mark start and end
                if len(trial_data) > 0:
                    plt.scatter(trial_data['x'].iloc[0], trial_data['y'].iloc[0], 
                               c='green', s=100, marker='o', label='Start', zorder=5)
                    plt.scatter(trial_data['x'].iloc[-1], trial_data['y'].iloc[-1], 
                               c='red', s=100, marker='X', label='End', zorder=5)
                
                plt.xlabel('X Coordinate (pixels)')
                plt.ylabel('Y Coordinate (pixels)')
                plt.title(f'{self.block_id} - Trial {trial_num} ({bodypart})')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                
                plot_file = plots_dir / f"{self.block_id}_trial_{trial_num:02d}_{bodypart}.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
        
        # Combined plot
        if save_combined:
            print(f"   📈 Creating combined plot with all {len(trials)} trials...")
            plt.figure(figsize=(12, 10))
            
            colors = plt.cm.viridis(np.linspace(0, 1, len(trials)))
            
            for i, trial_num in enumerate(trials):
                trial_data = trajectory_df[trajectory_df['trial_number'] == trial_num]
                plt.plot(trial_data['x'], trial_data['y'], 
                        color=colors[i], linewidth=1.5, alpha=0.7, 
                        label=f'Trial {trial_num}')
                
                # Mark start points
                if len(trial_data) > 0:
                    plt.scatter(trial_data['x'].iloc[0], trial_data['y'].iloc[0], 
                               color=colors[i], s=50, marker='o', alpha=0.8)
            
            plt.xlabel('X Coordinate (pixels)')
            plt.ylabel('Y Coordinate (pixels)')
            plt.title(f'{self.block_id} - All Trials ({bodypart})')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            
            # Legend handling
            if len(trials) <= 15:
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.legend(['Trial paths'], bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plot_file = plots_dir / f"{self.block_id}_all_trials_{bodypart}.png"
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"   ✅ Plots saved to: {plots_dir}")
    
    def create_metrics_plots(self) -> None:
        """
        Create plots for trial metrics (learning curves, distributions, etc.)
        """
        print(f"📊 Creating metrics plots for block {self.block_id}...")
        
        if self.trial_metrics is None:
            print("   ❌ No trial metrics available. Run compute_trial_metrics first.")
            return
        
        # Save plots directly to the main output directory (not in a subdirectory)
        plots_dir = self.output_dir
        
        # Key metrics to plot - filter out columns with all None/NaN values
        metrics_to_plot = []
        for col in self.trial_metrics.columns:
            if col not in ['block_id', 'trial_number']:
                # Check if column has any non-null values and is numeric-like
                non_null_values = self.trial_metrics[col].dropna()
                if len(non_null_values) > 0:
                    # Try to convert to numeric if it's timedelta
                    if pd.api.types.is_timedelta64_dtype(self.trial_metrics[col]):
                        metrics_to_plot.append(col)
                    elif self.trial_metrics[col].dtype in ['float64', 'int64']:
                        metrics_to_plot.append(col)
        
        # Convert timedelta columns to seconds for plotting
        plot_data = self.trial_metrics.copy()
        for col in plot_data.columns:
            if pd.api.types.is_timedelta64_dtype(plot_data[col]):
                plot_data[col] = plot_data[col].dt.total_seconds()
        
        # Individual metric plots
        for metric in metrics_to_plot:
            if metric in plot_data.columns and plot_data[metric].notna().any():
                plt.figure(figsize=(10, 6))
                
                # Get non-null data for plotting
                valid_data = plot_data[['trial_number', metric]].dropna()
                if len(valid_data) > 0:
                    plt.plot(valid_data['trial_number'], valid_data[metric], 'o-', linewidth=2, markersize=6)
                    plt.xlabel('Trial Number')
                    plt.ylabel(metric.replace('_', ' ').title())
                    plt.title(f'{self.block_id} - {metric.replace("_", " ").title()} Over Trials')
                    plt.grid(True, alpha=0.3)
                    
                    plot_file = plots_dir / f"{self.block_id}_metric_{metric}.png"
                    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
        
        # Combined metrics dashboard
        n_metrics = len(metrics_to_plot)
        if n_metrics > 0:
            cols = min(3, n_metrics)
            rows = (n_metrics + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if rows == 1 and cols == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            for i, metric in enumerate(metrics_to_plot):
                if i < len(axes) and metric in plot_data.columns:
                    # Get non-null data for plotting
                    valid_data = plot_data[['trial_number', metric]].dropna()
                    if len(valid_data) > 0:
                        axes[i].plot(valid_data['trial_number'], valid_data[metric], 'o-')
                        axes[i].set_xlabel('Trial Number')
                        axes[i].set_ylabel(metric.replace('_', ' ').title())
                        axes[i].set_title(metric.replace('_', ' ').title())
                        axes[i].grid(True, alpha=0.3)
            
            # Hide extra subplots
            for i in range(len(metrics_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.suptitle(f'{self.block_id} - Trial Metrics Dashboard', fontsize=16)
            plt.tight_layout()
            
            dashboard_file = plots_dir / f"{self.block_id}_metrics_dashboard.png"
            plt.savefig(dashboard_file, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"   ✅ Metrics plots saved to: {plots_dir}")
    
    def process_full_block(self, bodyparts: List[str] = None, create_plots: bool = True) -> Dict:
        """
        Run the complete processing pipeline for this block
        
        Args:
            bodyparts: List of bodyparts to analyze. If None, auto-detect.
            create_plots: Whether to generate visualization plots
        
        Returns:
            Dict with processing results
        """
        print(f"\n🧀 Processing full block: {self.block_id}")
        print("=" * 60)
        
        results = {'success': False, 'errors': []}
        
        try:
            # Step 1: Load data
            if not self.load_data():
                results['errors'].append("Failed to load data")
                return results
            
            # Step 2: Integrate data
            integrated = self.integrate_data()
            if integrated is None:
                results['errors'].append("Failed to integrate data")
                return results
            
            # Step 3: Segment trials
            segments = self.segment_trials()
            if segments is None:
                results['errors'].append("Failed to segment trials")
                return results
            
            # Step 4: Compute metrics
            metrics = self.compute_trial_metrics()
            if metrics is None:
                results['errors'].append("Failed to compute metrics")
                return results
            
            # Step 5: Extract trajectories
            trajectories = self.extract_bodypart_trajectories(bodyparts)
            
            # Step 6: Create plots
            if create_plots:
                if trajectories:
                    self.create_trial_plots(list(trajectories.keys())[0])
                self.create_metrics_plots()
            
            # Prepare results
            results.update({
                'success': True,
                'num_trials': len(segments),
                'num_bodyparts': len(trajectories),
                'bodyparts': list(trajectories.keys()),
                'output_directory': str(self.output_dir),
                'files_created': list(self.output_dir.glob("*"))
            })
            
            print("=" * 60)
            print(f"✅ Block processing complete!")
            print(f"   📊 Processed {results['num_trials']} trials")
            print(f"   🎯 Analyzed {results['num_bodyparts']} bodyparts: {results['bodyparts']}")
            print(f"   📁 Results saved to: {self.output_dir}")
            
        except Exception as e:
            results['errors'].append(f"Unexpected error: {e}")
            print(f"❌ Processing failed: {e}")
        
        return results
    
    def summary(self) -> None:
        """Print a summary of the processed block"""
        print(f"\n📋 Summary for block: {self.block_id}")
        print("-" * 40)
        
        if self.trial_segments is not None:
            print(f"Trials: {len(self.trial_segments)}")
        
        if self.trial_metrics is not None:
            # Summary statistics for key metrics that have data
            available_metrics = []
            for col in ['trial_length', 'first_reward_time', 'reward_collection_duration', 'num_rewards_collected']:
                if col in self.trial_metrics.columns:
                    values = self.trial_metrics[col].dropna()
                    if len(values) > 0:
                        available_metrics.append(col)
                        if pd.api.types.is_timedelta64_dtype(values):
                            values = values.dt.total_seconds()
                        if len(values) > 0:
                            print(f"{col}: {values.mean():.2f} ± {values.std():.2f} {'seconds' if 'time' in col or 'duration' in col else 'count'}")
            
            if not available_metrics:
                print("No complete metrics available (trials may be missing end markers or rewards)")
        
        if self.bodypart_data:
            print(f"Bodyparts analyzed: {list(self.bodypart_data.keys())}")
        
        print(f"Output directory: {self.output_dir}")


def process_single_block(block_id: str, timestamps_file: str, dlc_file: str, 
                        output_dir: str = None, bodyparts: List[str] = None) -> CheeseboardBlock:
    """
    Convenience function to process a single block with minimal setup
    
    Args:
        block_id: Unique identifier for the block
        timestamps_file: Path to timestamp CSV
        dlc_file: Path to DLC CSV
        output_dir: Output directory (optional)
        bodyparts: List of bodyparts to analyze (optional)
    
    Returns:
        CheeseboardBlock: Processed block object
    """
    block = CheeseboardBlock(block_id, timestamps_file, dlc_file, output_dir)
    block.process_full_block(bodyparts=bodyparts)
    return block


if __name__ == "__main__":
    print("🧀 Single Block Processor for Cheeseboard Analysis")
    print("=" * 50)
    
    # Example usage - update these paths for your data
    example_block_id = "ExperimentVideo_2025-08-19_1013"
    example_timestamps = r"C:\DATA\NICK Cheeseboard\Timestamps\ExperimentVideo_2025-08-19_1013_timestamps.csv"
    example_dlc = r"C:\DATA\NICK Cheeseboard\Videos\ExperimentVideo_2025-08-19_1013DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
    example_output = r"C:\DATA\NICK Cheeseboard\Single_Block_Output"
    
    print("Example usage:")
    print(f"block = process_single_block('{example_block_id}', timestamps_file, dlc_file, output_dir)")
    print("\nTo run the example, uncomment the following line:")
    print("# block = process_single_block(example_block_id, example_timestamps, example_dlc, example_output)")
    
    # Uncomment to run:
    block = process_single_block(example_block_id, example_timestamps, example_dlc, example_output)
    
    # If you're having issues with bodypart detection, inspect the DLC structure:
    # block.inspect_dlc_structure()
    
    block.summary()
