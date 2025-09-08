#!/usr/bin/env python3
"""
Migration utility to convert from old pipeline to new single_block_processor
Compares outputs to ensure the conversion preserves your analysis results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from single_block_processor import CheeseboardBlock

def compare_old_vs_new_pipeline(block_id: str, data_dir: Path):
    """
    Compare outputs from old pipeline vs new single_block_processor
    
    Args:
        block_id: The experiment block ID (e.g., 'ExperimentVideo_2025-08-13_1105')
        data_dir: Directory containing the raw data files
    """
    
    print(f"🔍 Comparing old vs new pipeline for: {block_id}")
    
    # File paths
    timestamps_file = data_dir / "RECORDED DATA" / f"{block_id}_timestamps.csv"
    dlc_file = data_dir / "RECORDED DATA" / f"{block_id}DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
    
    # Old pipeline outputs (if they exist)
    old_preprocessed = data_dir / "PREPROCESSED" / f"{block_id}_preprocessed.csv"
    old_split_by_trial = data_dir / "PREPROCESSED" / f"{block_id}_split_by_trial.csv"
    
    # New pipeline output directory
    new_output_dir = data_dir / "COMPARISON_OUTPUT" / block_id
    
    print(f"📂 Looking for old outputs:")
    print(f"   - Preprocessed: {old_preprocessed} {'✅' if old_preprocessed.exists() else '❌'}")
    print(f"   - Split by trial: {old_split_by_trial} {'✅' if old_split_by_trial.exists() else '❌'}")
    
    # Process with new pipeline
    print(f"🔄 Running new pipeline...")
    new_block = CheeseboardBlock(
        block_id=block_id,
        timestamps_file=str(timestamps_file),
        dlc_file=str(dlc_file),
        output_dir=str(new_output_dir)
    )
    
    # Process the block
    result = new_block.process_full_block(create_plots=False)
    
    if not result['success']:
        print(f"❌ New pipeline failed: {result['errors']}")
        return
    
    print(f"✅ New pipeline completed successfully")
    
    # Compare outputs if old ones exist
    if old_preprocessed.exists():
        compare_integrated_data(old_preprocessed, new_block)
    
    if old_split_by_trial.exists():
        compare_trial_segments(old_split_by_trial, new_block)
    
    # Generate comparison report
    generate_comparison_report(new_block, new_output_dir)

def compare_integrated_data(old_file: Path, new_block: CheeseboardBlock):
    """Compare the integrated data from old and new pipelines"""
    
    print("\n📊 Comparing integrated data...")
    
    try:
        old_data = pd.read_csv(old_file)
        new_data = new_block.integrated_data
        
        print(f"   Old data shape: {old_data.shape}")
        print(f"   New data shape: {new_data.shape}")
        
        # Compare timestamp columns
        if 'UnixTime' in old_data.columns and 'UnixTime' in new_data.columns:
            old_times = old_data['UnixTime'].iloc[2:]  # Skip header rows
            new_times = new_data['UnixTime'].iloc[2:]  # Skip header rows
            
            if len(old_times) == len(new_times):
                time_diff = np.abs(pd.to_numeric(old_times, errors='coerce') - 
                                 pd.to_numeric(new_times, errors='coerce')).sum()
                print(f"   Timestamp difference: {time_diff} {'✅' if time_diff < 1e-6 else '⚠️'}")
            else:
                print(f"   ⚠️  Different number of timestamp rows: old={len(old_times)}, new={len(new_times)}")
        
        print("   ✅ Integrated data comparison complete")
        
    except Exception as e:
        print(f"   ❌ Error comparing integrated data: {e}")

def compare_trial_segments(old_file: Path, new_block: CheeseboardBlock):
    """Compare trial segmentation from old and new pipelines"""
    
    print("\n🔍 Comparing trial segments...")
    
    try:
        old_segments = pd.read_csv(old_file)
        new_segments = new_block.trial_segments
        
        print(f"   Old segments: {len(old_segments)} trials")
        print(f"   New segments: {len(new_segments)} trials")
        
        if len(old_segments) == len(new_segments):
            print("   ✅ Same number of trials detected")
            
            # Compare start indices if available
            if 'Start_idx' in old_segments.columns and 'Start_idx' in new_segments.columns:
                start_diff = (old_segments['Start_idx'] - new_segments['Start_idx']).abs().sum()
                print(f"   Start index difference: {start_diff} {'✅' if start_diff == 0 else '⚠️'}")
            
        else:
            print(f"   ⚠️  Different number of trials detected")
        
        # Display first few trials for manual inspection
        print("\n   📋 First 3 trials comparison:")
        print("   Old format:")
        if len(old_segments) > 0:
            print(old_segments.head(3)[['Start_Duration', 'SB_Duration', 'R1_Duration', 'End_Duration']])
        
        print("   New format:")
        if len(new_segments) > 0:
            print(new_segments.head(3)[['Start_Duration', 'SB_Duration', 'R1_Duration', 'End_Duration']])
        
    except Exception as e:
        print(f"   ❌ Error comparing trial segments: {e}")

def generate_comparison_report(block: CheeseboardBlock, output_dir: Path):
    """Generate a comprehensive comparison report"""
    
    print(f"\n📝 Generating comparison report...")
    
    report_file = output_dir / f"{block.block_id}_comparison_report.txt"
    
    with open(report_file, 'w') as f:
        f.write(f"Comparison Report for {block.block_id}\n")
        f.write("="*50 + "\n\n")
        
        # Block summary
        f.write("BLOCK PROCESSING SUMMARY\n")
        f.write("-"*25 + "\n")
        if block.trial_segments is not None:
            f.write(f"Number of trials: {len(block.trial_segments)}\n")
        if block.trial_metrics is not None:
            f.write(f"Metrics computed: {len(block.trial_metrics)}\n")
        if block.bodypart_data:
            f.write(f"Bodyparts analyzed: {list(block.bodypart_data.keys())}\n")
        
        # Trial details
        f.write("\nTRIAL DETAILS\n")
        f.write("-"*15 + "\n")
        if block.trial_segments is not None:
            for _, trial in block.trial_segments.iterrows():
                f.write(f"Trial {trial.get('trial_number', '?')}: ")
                events = []
                for event in ['Start', 'SB', 'R1', 'R2', 'R3', 'End']:
                    if f"{event}_idx" in trial and pd.notna(trial[f"{event}_idx"]):
                        events.append(event)
                f.write(f"Events: {', '.join(events)}\n")
        
        # Metrics summary
        f.write("\nMETRICS SUMMARY\n")
        f.write("-"*15 + "\n")
        if block.trial_metrics is not None:
            for col in block.trial_metrics.columns:
                if col not in ['block_id', 'trial_number']:
                    values = block.trial_metrics[col].dropna()
                    if len(values) > 0:
                        if pd.api.types.is_timedelta64_dtype(values):
                            values_sec = values.dt.total_seconds()
                            f.write(f"{col}: {values_sec.mean():.2f} ± {values_sec.std():.2f} seconds\n")
                        elif pd.api.types.is_numeric_dtype(values):
                            f.write(f"{col}: {values.mean():.2f} ± {values.std():.2f}\n")
        
        f.write(f"\nFiles generated:\n")
        for file in output_dir.glob("*"):
            if file.is_file():
                f.write(f"  - {file.name}\n")
    
    print(f"   📄 Report saved: {report_file}")

def migrate_all_existing_data():
    """
    Migrate all existing data from old pipeline format to new format
    This will process all blocks in your RECORDED DATA directory
    """
    
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA")
    recorded_data_dir = data_dir / "RECORDED DATA"
    migration_output_dir = data_dir / "MIGRATED_OUTPUT"
    
    print(f"🔄 Starting migration of all existing data...")
    print(f"📂 Source: {recorded_data_dir}")
    print(f"📂 Output: {migration_output_dir}")
    
    # Find all timestamp files
    timestamp_files = list(recorded_data_dir.glob("*_timestamps.csv"))
    
    print(f"🔍 Found {len(timestamp_files)} experiments to migrate")
    
    migration_summary = []
    
    for i, ts_file in enumerate(timestamp_files):
        block_id = ts_file.stem.replace("_timestamps", "")
        dlc_file = recorded_data_dir / f"{block_id}DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
        
        print(f"\n📊 Migrating {i+1}/{len(timestamp_files)}: {block_id}")
        
        if not dlc_file.exists():
            print(f"   ⚠️  Skipping - DLC file not found")
            migration_summary.append({
                'block_id': block_id,
                'status': 'skipped',
                'reason': 'DLC file not found'
            })
            continue
        
        try:
            # Process with new pipeline
            block = CheeseboardBlock(
                block_id=block_id,
                timestamps_file=str(ts_file),
                dlc_file=str(dlc_file),
                output_dir=str(migration_output_dir / block_id)
            )
            
            result = block.process_full_block()
            
            if result['success']:
                migration_summary.append({
                    'block_id': block_id,
                    'status': 'success',
                    'num_trials': result['num_trials'],
                    'bodyparts': result['bodyparts']
                })
                print(f"   ✅ Successfully migrated ({result['num_trials']} trials)")
            else:
                migration_summary.append({
                    'block_id': block_id,
                    'status': 'failed',
                    'errors': result['errors']
                })
                print(f"   ❌ Failed: {result['errors']}")
                
        except Exception as e:
            migration_summary.append({
                'block_id': block_id,
                'status': 'error',
                'error': str(e)
            })
            print(f"   ❌ Error: {e}")
    
    # Generate migration summary report
    summary_df = pd.DataFrame(migration_summary)
    summary_file = migration_output_dir / "migration_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\n📈 Migration Summary:")
    print(f"   Total experiments: {len(migration_summary)}")
    print(f"   Successful: {len(summary_df[summary_df['status'] == 'success'])}")
    print(f"   Failed: {len(summary_df[summary_df['status'] == 'failed'])}")
    print(f"   Skipped: {len(summary_df[summary_df['status'] == 'skipped'])}")
    print(f"   📄 Summary saved: {summary_file}")

if __name__ == "__main__":
    print("🔄 Pipeline Migration and Comparison Tool")
    print("="*50)
    
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA")
    
    # Test with one specific block
    test_block = "ExperimentVideo_2025-08-13_1105"
    print(f"🧪 Testing migration with: {test_block}")
    compare_old_vs_new_pipeline(test_block, data_dir)
    
    # Uncomment to migrate all data
    # print("\n" + "="*50)
    # print("🔄 Migrating all existing data...")
    # migrate_all_existing_data()
