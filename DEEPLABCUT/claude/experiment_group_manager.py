#!/usr/bin/env python3
"""
Experiment Group Manager
Creates templates for manually assigning experiments to groups and marking invalid trials
"""

import pandas as pd
from pathlib import Path
import csv
from datetime import datetime

def create_experiment_groups_template(batch_output_dir: str, template_file: str = None):
    """
    Create a CSV template for manually assigning experiments to groups
    
    Args:
        batch_output_dir: Directory containing processed experiments
        template_file: Output template file path
    """
    
    batch_dir = Path(batch_output_dir)
    if not batch_dir.exists():
        print(f"❌ Batch directory not found: {batch_dir}")
        return None
    
    if template_file is None:
        template_file = batch_dir / "experiment_groups_template.csv"
    
    print(f"📋 Creating experiment groups template...")
    print(f"📂 Source: {batch_dir}")
    print(f"📄 Template: {template_file}")
    
    # Find all experiment directories
    exp_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith('ExperimentVideo_')]
    exp_dirs = sorted(exp_dirs)
    
    experiments_data = []
    
    for exp_dir in exp_dirs:
        block_id = exp_dir.name
        
        # Extract date and time
        parts = block_id.split('_')
        date = parts[1] if len(parts) > 1 else 'unknown'
        time = parts[2] if len(parts) > 2 else 'unknown'
        
        # Load trial metrics to get trial count
        metrics_file = exp_dir / f"{block_id}_trial_metrics.csv"
        num_trials = 0
        if metrics_file.exists():
            try:
                metrics_df = pd.read_csv(metrics_file)
                num_trials = len(metrics_df)
            except:
                pass
        
        # Get basic info
        experiments_data.append({
            'block_id': block_id,
            'experiment_date': date,
            'experiment_time': time,
            'num_trials': num_trials,
            'experiment_group': '',  # TO BE FILLED BY USER
            'notes': '',  # Optional notes
            'include_in_analysis': 'yes'  # yes/no
        })
    
    # Create DataFrame and save
    template_df = pd.DataFrame(experiments_data)
    template_df.to_csv(template_file, index=False)
    
    print(f"\n✅ Experiment groups template created!")
    print(f"📊 Found {len(experiments_data)} experiments")
    print(f"💾 Template saved: {template_file}")
    
    # Show instructions
    print(f"\n📝 INSTRUCTIONS:")
    print(f"1. Open: {template_file}")
    print(f"2. Fill in the 'experiment_group' column with your group names")
    print(f"   Examples: 'Control', 'Treatment', 'Baseline', 'Drug_A', etc.")
    print(f"3. Set 'include_in_analysis' to 'no' for experiments to exclude")
    print(f"4. Add any notes in the 'notes' column")
    print(f"5. Save the file and use it with the analysis scripts")
    
    return template_df

def create_trial_exclusion_template(batch_output_dir: str, template_file: str = None):
    """
    Create a CSV template for manually marking trials to exclude
    
    Args:
        batch_output_dir: Directory containing processed experiments  
        template_file: Output template file path
    """
    
    batch_dir = Path(batch_output_dir)
    if not batch_dir.exists():
        print(f"❌ Batch directory not found: {batch_dir}")
        return None
    
    if template_file is None:
        template_file = batch_dir / "trial_exclusions_template.csv"
    
    print(f"📋 Creating trial exclusions template...")
    print(f"📂 Source: {batch_dir}")
    print(f"📄 Template: {template_file}")
    
    # Find all experiment directories
    exp_dirs = [d for d in batch_dir.iterdir() if d.is_dir() and d.name.startswith('ExperimentVideo_')]
    exp_dirs = sorted(exp_dirs)
    
    trial_data = []
    
    for exp_dir in exp_dirs:
        block_id = exp_dir.name
        
        # Extract date and time
        parts = block_id.split('_')
        date = parts[1] if len(parts) > 1 else 'unknown'
        time = parts[2] if len(parts) > 2 else 'unknown'
        
        # Load trial metrics
        metrics_file = exp_dir / f"{block_id}_trial_metrics.csv"
        if not metrics_file.exists():
            continue
            
        try:
            metrics_df = pd.read_csv(metrics_file)
            
            for _, trial_row in metrics_df.iterrows():
                trial_data.append({
                    'block_id': block_id,
                    'experiment_date': date,
                    'experiment_time': time,
                    'trial_number': trial_row['trial_number'],
                    'first_reward_time_sec': trial_row.get('first_reward_time', '').total_seconds() if pd.notna(trial_row.get('first_reward_time')) else '',
                    'num_rewards_collected': trial_row.get('num_rewards_collected', ''),
                    'exclude_trial': 'no',  # TO BE FILLED BY USER (yes/no)
                    'exclusion_reason': ''  # Optional reason for exclusion
                })
                
        except Exception as e:
            print(f"   ⚠️  Error reading {metrics_file}: {e}")
            continue
    
    # Create DataFrame and save
    template_df = pd.DataFrame(trial_data)
    template_df.to_csv(template_file, index=False)
    
    print(f"\n✅ Trial exclusions template created!")
    print(f"📊 Found {len(trial_data)} trials across all experiments")
    print(f"💾 Template saved: {template_file}")
    
    # Show instructions
    print(f"\n📝 INSTRUCTIONS:")
    print(f"1. Open: {template_file}")
    print(f"2. Set 'exclude_trial' to 'yes' for trials to exclude from analysis")
    print(f"3. Add reasons in 'exclusion_reason' column (optional)")
    print(f"   Examples: 'poor_tracking', 'equipment_error', 'animal_distressed', etc.")
    print(f"4. Save the file and use it with the analysis scripts")
    
    return template_df

def load_experiment_groups(groups_file: str):
    """Load experiment groups from CSV file"""
    
    groups_path = Path(groups_file)
    if not groups_path.exists():
        print(f"❌ Groups file not found: {groups_file}")
        return None
    
    try:
        groups_df = pd.read_csv(groups_file)
        
        # Validate required columns
        required_cols = ['block_id', 'experiment_group', 'include_in_analysis']
        missing_cols = [col for col in required_cols if col not in groups_df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Filter to only included experiments
        included_df = groups_df[groups_df['include_in_analysis'].str.lower() == 'yes'].copy()
        
        print(f"✅ Loaded experiment groups:")
        print(f"   Total experiments: {len(groups_df)}")
        print(f"   Included experiments: {len(included_df)}")
        
        # Show group summary
        group_summary = included_df['experiment_group'].value_counts()
        print(f"   Groups: {dict(group_summary)}")
        
        return included_df
        
    except Exception as e:
        print(f"❌ Error loading groups file: {e}")
        return None

def load_trial_exclusions(exclusions_file: str):
    """Load trial exclusions from CSV file"""
    
    exclusions_path = Path(exclusions_file)
    if not exclusions_path.exists():
        print(f"❌ Exclusions file not found: {exclusions_file}")
        return None
    
    try:
        exclusions_df = pd.read_csv(exclusions_file)
        
        # Validate required columns
        required_cols = ['block_id', 'trial_number', 'exclude_trial']
        missing_cols = [col for col in required_cols if col not in exclusions_df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return None
        
        # Filter to only excluded trials
        excluded_df = exclusions_df[exclusions_df['exclude_trial'].str.lower() == 'yes'].copy()
        
        print(f"✅ Loaded trial exclusions:")
        print(f"   Total trials: {len(exclusions_df)}")
        print(f"   Excluded trials: {len(excluded_df)}")
        
        if len(excluded_df) > 0:
            # Show exclusion summary
            exclusion_summary = excluded_df['exclusion_reason'].value_counts()
            print(f"   Exclusion reasons: {dict(exclusion_summary)}")
        
        return excluded_df
        
    except Exception as e:
        print(f"❌ Error loading exclusions file: {e}")
        return None

if __name__ == "__main__":
    print("🧀 Experiment Group Manager")
    print("="*50)
    
    # Default paths
    batch_dir = "/Users/nick/Projects/cheeseboardAnalysis/DATA/BATCH_ANALYSIS_ALL_DAYS"
    
    print("🔧 Creating templates...")
    
    # Create experiment groups template
    exp_groups_template = create_experiment_groups_template(batch_dir)
    
    # Create trial exclusions template  
    trial_exclusions_template = create_trial_exclusion_template(batch_dir)
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Fill out the experiment groups template")
    print(f"2. Fill out the trial exclusions template")
    print(f"3. Use these files with the group analysis scripts")
    
    # Test loading (will show errors since templates are empty)
    print(f"\n🧪 Testing template loading...")
    groups_file = f"{batch_dir}/experiment_groups_template.csv"
    exclusions_file = f"{batch_dir}/trial_exclusions_template.csv"
    
    print("   (These will show as empty until you fill them out)")
