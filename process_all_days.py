#!/usr/bin/env python3
"""
Batch processor to analyze ALL available experimental days
Processes every experiment in your dataset and creates comprehensive comparisons
"""

import pandas as pd
from pathlib import Path
from single_block_processor import CheeseboardBlock
import time
from datetime import datetime

def find_all_experiments():
    """Find all available experiments with both timestamp and DLC files"""
    
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA")
    timestamp_files = sorted(list(data_dir.glob("*_timestamps.csv")))
    
    experiments = []
    
    print("🔍 Scanning for available experiments...")
    print("="*60)
    
    for ts_file in timestamp_files:
        block_id = ts_file.stem.replace("_timestamps", "")
        dlc_file = data_dir / f"{block_id}DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
        
        # Get file info
        ts_size = ts_file.stat().st_size / (1024 * 1024)  # MB
        dlc_exists = dlc_file.exists()
        dlc_size = dlc_file.stat().st_size / (1024 * 1024) if dlc_exists else 0
        
        # Extract date and time info
        parts = block_id.split("_")
        date = parts[1] if len(parts) > 1 else "unknown"
        time_str = parts[2] if len(parts) > 2 else "unknown"
        
        status = "✅" if dlc_exists else "❌"
        print(f"{status} {block_id}")
        print(f"    📅 {date} at {time_str}")
        print(f"    📊 Timestamps: {ts_size:.1f} MB, DLC: {dlc_size:.1f} MB")
        
        if dlc_exists:
            experiments.append({
                'block_id': block_id,
                'date': date,
                'time': time_str,
                'ts_file': ts_file,
                'dlc_file': dlc_file,
                'ts_size_mb': ts_size,
                'dlc_size_mb': dlc_size
            })
        print()
    
    print(f"📊 Found {len(experiments)} complete experiments to process")
    return experiments

def process_all_experiments():
    """Process all available experiments"""
    
    experiments = find_all_experiments()
    
    if not experiments:
        print("❌ No complete experiments found!")
        return
    
    # Set up batch output directory
    batch_output_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/BATCH_ANALYSIS_ALL_DAYS")
    batch_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🚀 Starting batch processing of {len(experiments)} experiments")
    print(f"📁 Batch output directory: {batch_output_dir}")
    print("="*60)
    
    # Track processing results
    results = []
    start_time = time.time()
    
    for i, exp in enumerate(experiments):
        exp_start_time = time.time()
        block_id = exp['block_id']
        
        print(f"\n📊 Processing {i+1}/{len(experiments)}: {block_id}")
        print("-" * 50)
        
        try:
            # Create individual experiment output directory
            exp_output_dir = batch_output_dir / block_id
            
            # Process the experiment
            block = CheeseboardBlock(
                block_id=block_id,
                timestamps_file=str(exp['ts_file']),
                dlc_file=str(exp['dlc_file']),
                output_dir=str(exp_output_dir)
            )
            
            # Run full processing
            result = block.process_full_block(create_plots=True)
            
            exp_time = time.time() - exp_start_time
            
            if result['success']:
                print(f"   ✅ Success! ({exp_time:.1f} seconds)")
                print(f"      📊 Trials: {result['num_trials']}")
                print(f"      🎯 Bodyparts: {result['num_bodyparts']}")
                
                # Store results for summary
                results.append({
                    'block_id': block_id,
                    'date': exp['date'],
                    'time': exp['time'],
                    'status': 'success',
                    'num_trials': result['num_trials'],
                    'num_bodyparts': result['num_bodyparts'],
                    'bodyparts': ', '.join(result['bodyparts']),
                    'processing_time_sec': exp_time,
                    'data_size_mb': exp['ts_size_mb'] + exp['dlc_size_mb'],
                    'output_directory': str(exp_output_dir)
                })
                
            else:
                print(f"   ❌ Failed! ({exp_time:.1f} seconds)")
                print(f"      Errors: {result['errors']}")
                
                results.append({
                    'block_id': block_id,
                    'date': exp['date'],
                    'time': exp['time'],
                    'status': 'failed',
                    'errors': str(result['errors']),
                    'processing_time_sec': exp_time,
                    'data_size_mb': exp['ts_size_mb'] + exp['dlc_size_mb']
                })
                
        except Exception as e:
            exp_time = time.time() - exp_start_time
            print(f"   💥 Exception! ({exp_time:.1f} seconds)")
            print(f"      Error: {e}")
            
            results.append({
                'block_id': block_id,
                'date': exp['date'],
                'time': exp['time'],
                'status': 'exception',
                'error': str(e),
                'processing_time_sec': exp_time,
                'data_size_mb': exp['ts_size_mb'] + exp['dlc_size_mb']
            })
    
    total_time = time.time() - start_time
    
    # Save processing summary
    results_df = pd.DataFrame(results)
    summary_file = batch_output_dir / "batch_processing_summary.csv"
    results_df.to_csv(summary_file, index=False)
    
    # Print final summary
    print(f"\n" + "="*60)
    print("🎉 BATCH PROCESSING COMPLETE!")
    print("="*60)
    
    successful = len(results_df[results_df['status'] == 'success'])
    failed = len(results_df[results_df['status'] == 'failed'])
    exceptions = len(results_df[results_df['status'] == 'exception'])
    
    print(f"📊 Processing Summary:")
    print(f"   ✅ Successful: {successful}")
    print(f"   ❌ Failed: {failed}")
    print(f"   💥 Exceptions: {exceptions}")
    print(f"   ⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   📄 Summary saved: {summary_file}")
    
    # Show successful experiments
    if successful > 0:
        success_df = results_df[results_df['status'] == 'success']
        total_trials = success_df['num_trials'].sum()
        avg_trials = success_df['num_trials'].mean()
        
        print(f"\n📈 Successful Experiments Detail:")
        print(f"   🔢 Total trials across all experiments: {total_trials}")
        print(f"   📊 Average trials per experiment: {avg_trials:.1f}")
        
        print(f"\n📅 By Date:")
        for date in sorted(success_df['date'].unique()):
            date_exps = success_df[success_df['date'] == date]
            date_trials = date_exps['num_trials'].sum()
            print(f"   {date}: {len(date_exps)} experiments, {date_trials} trials")
    
    return results_df, batch_output_dir

def create_cross_experiment_analysis(results_df, batch_output_dir):
    """Create comprehensive analysis across all experiments"""
    
    if len(results_df[results_df['status'] == 'success']) == 0:
        print("⚠️  No successful experiments to analyze")
        return
    
    print(f"\n🔬 Creating cross-experiment analysis...")
    
    # Collect all trial metrics from successful experiments
    all_metrics = []
    all_trajectories = {}
    
    success_experiments = results_df[results_df['status'] == 'success']
    
    for _, exp in success_experiments.iterrows():
        exp_dir = Path(exp['output_directory'])
        
        # Load trial metrics
        metrics_file = exp_dir / f"{exp['block_id']}_trial_metrics.csv"
        if metrics_file.exists():
            metrics = pd.read_csv(metrics_file)
            metrics['experiment_date'] = exp['date']
            metrics['experiment_time'] = exp['time']
            metrics['experiment_id'] = exp['block_id']
            all_metrics.append(metrics)
        
        # Collect trajectory info (just metadata for now)
        trajectory_files = list(exp_dir.glob("*_trajectory.csv"))
        if trajectory_files:
            for traj_file in trajectory_files:
                bodypart = traj_file.stem.split('_')[-2]  # Extract bodypart name
                if bodypart not in all_trajectories:
                    all_trajectories[bodypart] = []
                all_trajectories[bodypart].append({
                    'experiment': exp['block_id'],
                    'date': exp['date'],
                    'file': traj_file
                })
    
    if not all_metrics:
        print("⚠️  No trial metrics found")
        return
    
    # Combine all metrics
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_file = batch_output_dir / "combined_all_experiments_metrics.csv"
    combined_metrics.to_csv(combined_file, index=False)
    
    print(f"   📊 Combined metrics from {len(success_experiments)} experiments")
    print(f"   🔢 Total trials: {len(combined_metrics)}")
    print(f"   📄 Saved: {combined_file}")
    
    # Create summary statistics
    print(f"\n📈 Cross-Experiment Summary Statistics:")
    
    numeric_cols = ['first_reward_time', 'trial_length', 'reward_collection_duration', 'num_rewards_collected']
    
    for col in numeric_cols:
        if col in combined_metrics.columns:
            values = combined_metrics[col].dropna()
            if len(values) > 0:
                if pd.api.types.is_timedelta64_dtype(values):
                    values_sec = values.dt.total_seconds()
                    print(f"   {col}: {values_sec.mean():.2f} ± {values_sec.std():.2f} seconds (n={len(values)})")
                else:
                    print(f"   {col}: {values.mean():.2f} ± {values.std():.2f} (n={len(values)})")
    
    # Group by date
    print(f"\n📅 Performance by Date:")
    date_summary = combined_metrics.groupby('experiment_date').agg({
        'first_reward_time': lambda x: x.dt.total_seconds().mean() if len(x.dropna()) > 0 else None,
        'num_rewards_collected': 'mean',
        'trial_number': 'count'
    }).round(2)
    date_summary.columns = ['Avg_First_Reward_Sec', 'Avg_Rewards', 'Total_Trials']
    print(date_summary)
    
    # Save date summary
    date_summary_file = batch_output_dir / "summary_by_date.csv"
    date_summary.to_csv(date_summary_file)
    
    print(f"\n🎯 Available bodyparts across experiments:")
    for bodypart, experiments in all_trajectories.items():
        print(f"   {bodypart}: {len(experiments)} experiments")
    
    print(f"\n✅ Cross-experiment analysis complete!")
    print(f"📁 All results saved in: {batch_output_dir}")

if __name__ == "__main__":
    print("🧀 Batch Analysis - ALL EXPERIMENTAL DAYS")
    print("="*60)
    print(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Process all experiments
    results_df, batch_dir = process_all_experiments()
    
    # Create cross-experiment analysis
    create_cross_experiment_analysis(results_df, batch_dir)
    
    print(f"\n🎉 ALL DONE!")
    print(f"📁 Check your comprehensive results in:")
    print(f"   {batch_dir}")
    print(f"🕐 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
