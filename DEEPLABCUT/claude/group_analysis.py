#!/usr/bin/env python3
"""
Group Analysis with Manual Labels
Analyzes experiments by manually assigned groups and excludes marked trials
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add claude folder to path
sys.path.append(str(Path(__file__).parent))
from experiment_group_manager import load_experiment_groups, load_trial_exclusions

def load_and_filter_data(batch_dir: str, groups_file: str, exclusions_file: str = None):
    """
    Load all experiment data and apply group/exclusion filters
    
    Args:
        batch_dir: Directory with processed experiments
        groups_file: CSV file with experiment group assignments
        exclusions_file: CSV file with trial exclusions (optional)
    
    Returns:
        Dict with filtered metrics and trajectory data
    """
    
    print(f"📂 Loading and filtering data...")
    
    # Load group assignments
    groups_df = load_experiment_groups(groups_file)
    if groups_df is None:
        return None
    
    # Load trial exclusions (optional)
    exclusions_df = None
    if exclusions_file and Path(exclusions_file).exists():
        exclusions_df = load_trial_exclusions(exclusions_file)
    
    # Get list of included experiments
    included_experiments = groups_df['block_id'].tolist()
    
    print(f"🔍 Processing {len(included_experiments)} included experiments...")
    
    # Load metrics for all included experiments
    all_metrics = []
    batch_path = Path(batch_dir)
    
    for block_id in included_experiments:
        exp_dir = batch_path / block_id
        metrics_file = exp_dir / f"{block_id}_trial_metrics.csv"
        
        if not metrics_file.exists():
            print(f"   ⚠️  No metrics file for {block_id}")
            continue
        
        try:
            metrics_df = pd.read_csv(metrics_file)
            
            # Add experiment metadata
            exp_info = groups_df[groups_df['block_id'] == block_id].iloc[0]
            metrics_df['block_id'] = block_id
            metrics_df['experiment_group'] = exp_info['experiment_group']
            metrics_df['experiment_date'] = exp_info['experiment_date']
            metrics_df['experiment_time'] = exp_info['experiment_time']
            
            # Apply trial exclusions if provided
            if exclusions_df is not None:
                exp_exclusions = exclusions_df[exclusions_df['block_id'] == block_id]
                excluded_trials = exp_exclusions['trial_number'].tolist()
                
                if excluded_trials:
                    print(f"   🚫 Excluding {len(excluded_trials)} trials from {block_id}")
                    metrics_df = metrics_df[~metrics_df['trial_number'].isin(excluded_trials)]
            
            all_metrics.append(metrics_df)
            
        except Exception as e:
            print(f"   ❌ Error loading {block_id}: {e}")
            continue
    
    if not all_metrics:
        print("❌ No valid metrics data found!")
        return None
    
    # Combine all metrics
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    
    print(f"✅ Data loading complete!")
    print(f"   📊 Total trials: {len(combined_metrics)}")
    print(f"   🧪 Experiments: {combined_metrics['block_id'].nunique()}")
    print(f"   🏷️  Groups: {combined_metrics['experiment_group'].nunique()}")
    
    # Show group breakdown
    group_summary = combined_metrics.groupby('experiment_group').agg({
        'block_id': 'nunique',
        'trial_number': 'count'
    }).round(2)
    group_summary.columns = ['Experiments', 'Trials']
    print(f"\n📈 Group Summary:")
    print(group_summary)
    
    return {
        'metrics': combined_metrics,
        'groups_info': groups_df,
        'exclusions_info': exclusions_df
    }

def create_group_comparison_plots(data_dict: dict, output_dir: str):
    """
    Create comparison plots between experiment groups
    
    Args:
        data_dict: Dictionary with filtered data from load_and_filter_data
        output_dir: Directory to save plots
    """
    
    metrics_df = data_dict['metrics']
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 Creating group comparison plots...")
    print(f"📁 Output directory: {output_path}")
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Key metrics to plot
    metric_configs = {
        'first_reward_time': {
            'title': 'Time to First Reward',
            'ylabel': 'Time (seconds)',
            'convert_func': lambda x: x.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(x) else x
        },
        'trial_length': {
            'title': 'Trial Length',
            'ylabel': 'Duration (seconds)', 
            'convert_func': lambda x: x.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(x) else x
        },
        'num_rewards_collected': {
            'title': 'Number of Rewards Collected',
            'ylabel': 'Count',
            'convert_func': lambda x: x
        },
        'reward_collection_duration': {
            'title': 'Reward Collection Duration',
            'ylabel': 'Duration (seconds)',
            'convert_func': lambda x: x.dt.total_seconds() if pd.api.types.is_timedelta64_dtype(x) else x
        }
    }
    
    # Create individual metric plots
    for metric, config in metric_configs.items():
        if metric not in metrics_df.columns:
            continue
        
        # Convert data if needed
        plot_data = metrics_df.copy()
        plot_data[metric] = config['convert_func'](plot_data[metric])
        plot_data = plot_data.dropna(subset=[metric])
        
        if len(plot_data) == 0:
            continue
        
        # Box plot comparison
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=plot_data, x='experiment_group', y=metric)
        plt.title(f'{config["title"]} by Experiment Group')
        plt.ylabel(config['ylabel'])
        plt.xlabel('Experiment Group')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plot_file = output_path / f"group_comparison_{metric}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Learning curve by group
        plt.figure(figsize=(12, 6))
        
        for group in plot_data['experiment_group'].unique():
            group_data = plot_data[plot_data['experiment_group'] == group]
            
            # Calculate mean and SEM by trial number
            trial_summary = group_data.groupby('trial_number')[metric].agg(['mean', 'sem', 'count']).reset_index()
            trial_summary = trial_summary[trial_summary['count'] >= 2]  # Only trials with at least 2 data points
            
            plt.errorbar(trial_summary['trial_number'], trial_summary['mean'], 
                        yerr=trial_summary['sem'], label=group, marker='o', linewidth=2)
        
        plt.title(f'{config["title"]} Learning Curve by Group')
        plt.xlabel('Trial Number')
        plt.ylabel(config['ylabel'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        curve_file = output_path / f"learning_curve_{metric}.png"
        plt.savefig(curve_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Created plots for {metric}")
    
    # Summary statistics table
    print(f"\n📈 Creating summary statistics...")
    
    summary_stats = []
    
    for metric in metric_configs.keys():
        if metric not in metrics_df.columns:
            continue
        
        plot_data = metrics_df.copy()
        plot_data[metric] = metric_configs[metric]['convert_func'](plot_data[metric])
        plot_data = plot_data.dropna(subset=[metric])
        
        for group in plot_data['experiment_group'].unique():
            group_data = plot_data[plot_data['experiment_group'] == group][metric]
            
            summary_stats.append({
                'metric': metric,
                'group': group,
                'n_trials': len(group_data),
                'n_experiments': plot_data[plot_data['experiment_group'] == group]['block_id'].nunique(),
                'mean': group_data.mean(),
                'std': group_data.std(),
                'sem': group_data.sem(),
                'median': group_data.median()
            })
    
    # Save summary stats
    summary_df = pd.DataFrame(summary_stats)
    summary_file = output_path / "group_summary_statistics.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"   💾 Summary statistics saved: {summary_file}")
    
    # Create overall dashboard
    create_group_dashboard(data_dict, output_path)
    
    print(f"✅ Group comparison plots complete!")
    return output_path

def create_group_dashboard(data_dict: dict, output_dir: Path):
    """Create a comprehensive dashboard plot"""
    
    metrics_df = data_dict['metrics']
    
    # Prepare data
    plot_data = metrics_df.copy()
    
    # Convert time columns
    time_cols = ['first_reward_time', 'trial_length', 'reward_collection_duration']
    for col in time_cols:
        if col in plot_data.columns:
            if pd.api.types.is_timedelta64_dtype(plot_data[col]):
                plot_data[col] = plot_data[col].dt.total_seconds()
    
    # Create dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment Groups Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: First reward time
    if 'first_reward_time' in plot_data.columns:
        sns.boxplot(data=plot_data.dropna(subset=['first_reward_time']), 
                   x='experiment_group', y='first_reward_time', ax=axes[0,0])
        axes[0,0].set_title('Time to First Reward')
        axes[0,0].set_ylabel('Time (seconds)')
        axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Number of rewards
    if 'num_rewards_collected' in plot_data.columns:
        sns.boxplot(data=plot_data.dropna(subset=['num_rewards_collected']), 
                   x='experiment_group', y='num_rewards_collected', ax=axes[0,1])
        axes[0,1].set_title('Rewards Collected per Trial')
        axes[0,1].set_ylabel('Count')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Trial length
    if 'trial_length' in plot_data.columns:
        sns.boxplot(data=plot_data.dropna(subset=['trial_length']), 
                   x='experiment_group', y='trial_length', ax=axes[1,0])
        axes[1,0].set_title('Trial Duration')
        axes[1,0].set_ylabel('Duration (seconds)')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Sample sizes
    group_summary = plot_data.groupby('experiment_group').agg({
        'block_id': 'nunique',
        'trial_number': 'count'
    }).reset_index()
    
    x_pos = range(len(group_summary))
    axes[1,1].bar([i-0.2 for i in x_pos], group_summary['block_id'], 
                  width=0.4, label='Experiments', alpha=0.7)
    axes[1,1].bar([i+0.2 for i in x_pos], group_summary['trial_number'], 
                  width=0.4, label='Trials', alpha=0.7)
    axes[1,1].set_title('Sample Sizes by Group')
    axes[1,1].set_ylabel('Count')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(group_summary['experiment_group'], rotation=45)
    axes[1,1].legend()
    
    plt.tight_layout()
    
    dashboard_file = output_dir / "experiment_groups_dashboard.png"
    plt.savefig(dashboard_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   📊 Dashboard saved: {dashboard_file}")

if __name__ == "__main__":
    print("🧀 Group Analysis with Manual Labels")
    print("="*50)
    
    # Default paths - UPDATE THESE AFTER CREATING YOUR TEMPLATES
    batch_dir = "/Users/nick/Projects/cheeseboardAnalysis/DATA/BATCH_ANALYSIS_ALL_DAYS"
    groups_file = f"{batch_dir}/experiment_groups_template.csv"  # Fill this out first!
    exclusions_file = f"{batch_dir}/trial_exclusions_template.csv"  # Fill this out first!
    output_dir = "/Users/nick/Projects/cheeseboardAnalysis/DATA/GROUP_ANALYSIS"
    
    print("📋 Input files:")
    print(f"   Groups: {groups_file}")
    print(f"   Exclusions: {exclusions_file}")
    print(f"   Output: {output_dir}")
    
    # Check if template files exist and have been filled out
    if not Path(groups_file).exists():
        print(f"\n❌ Groups file not found!")
        print(f"First run: python experiment_group_manager.py")
        print(f"Then fill out the templates before running this script.")
    else:
        # Load and analyze data
        data_dict = load_and_filter_data(batch_dir, groups_file, exclusions_file)
        
        if data_dict is not None:
            # Create group comparison plots
            create_group_comparison_plots(data_dict, output_dir)
            
            print(f"\n🎉 Group analysis complete!")
            print(f"📁 Check results in: {output_dir}")
        else:
            print(f"\n❌ Data loading failed!")
            print(f"Check your groups file and make sure experiment groups are filled out.")
