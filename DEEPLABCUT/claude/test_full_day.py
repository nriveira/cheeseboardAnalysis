#!/usr/bin/env python3
"""
Comprehensive test with a full day of real data
Tests all functionality: integration, segmentation, metrics, trajectories, and plotting
"""

import pandas as pd
from pathlib import Path
from single_block_processor import CheeseboardBlock, process_single_block

def test_full_day_analysis():
    """Test with real data from one experimental day"""
    
    print("🧪 Testing Full Day Analysis with Real Data")
    print("="*60)
    
    # Set up paths for your data
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA")
    output_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/FULL_DAY_TEST")
    
    # Pick a specific day to test - let's use Aug 13 data
    block_id = "ExperimentVideo_2025-08-13_1105"
    timestamps_file = data_dir / f"{block_id}_timestamps.csv"
    dlc_file = data_dir / f"{block_id}DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
    
    print(f"📅 Testing with: {block_id}")
    print(f"📂 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    # Check if files exist
    if not timestamps_file.exists():
        print(f"❌ Timestamps file not found: {timestamps_file}")
        return None
    
    if not dlc_file.exists():
        print(f"❌ DLC file not found: {dlc_file}")
        return None
    
    print("✅ Files found, starting comprehensive analysis...")
    
    try:
        # Create block processor
        block = CheeseboardBlock(
            block_id=block_id,
            timestamps_file=str(timestamps_file),
            dlc_file=str(dlc_file),
            output_dir=str(output_dir)
        )
        
        print("\n" + "="*60)
        print("STEP 1: LOADING AND INSPECTING DATA")
        print("="*60)
        
        if not block.load_data():
            print("❌ Failed to load data")
            return None
        
        # Show data structure
        print(f"\n📊 Data Overview:")
        print(f"   Timestamp rows: {len(block.raw_timestamps)}")
        print(f"   DLC rows: {len(block.raw_dlc)}")
        print(f"   DLC columns: {list(block.raw_dlc.columns)}")
        
        # Inspect DLC structure
        print(f"\n🔍 DLC Structure Inspection:")
        block.inspect_dlc_structure()
        
        print("\n" + "="*60)
        print("STEP 2: DATA INTEGRATION")
        print("="*60)
        
        integrated = block.integrate_data()
        if integrated is None:
            print("❌ Failed to integrate data")
            return None
        
        print(f"\n📋 Integrated Data Preview:")
        print(integrated.head(10))
        print(f"\nIntegrated data shape: {integrated.shape}")
        
        print("\n" + "="*60)
        print("STEP 3: TRIAL SEGMENTATION")
        print("="*60)
        
        segments = block.segment_trials()
        if segments is None:
            print("❌ Failed to segment trials")
            return None
        
        print(f"\n📊 Trial Segmentation Results:")
        print(f"   Number of trials: {len(segments)}")
        print(f"\n📋 Trial Segments Detail:")
        for _, trial in segments.iterrows():
            events = []
            for event in ['Start', 'SB', 'R1', 'R2', 'R3', 'End']:
                if f"{event}_idx" in trial and pd.notna(trial[f"{event}_idx"]):
                    events.append(f"{event}({int(trial[f'{event}_idx'])})")
            print(f"   Trial {trial.get('trial_number', '?')}: {' -> '.join(events)}")
        
        print("\n" + "="*60)
        print("STEP 4: BEHAVIORAL METRICS")
        print("="*60)
        
        metrics = block.compute_trial_metrics()
        if metrics is None:
            print("❌ Failed to compute metrics")
            return None
        
        print(f"\n📊 Behavioral Metrics Summary:")
        print(metrics)
        
        # Show summary statistics
        print(f"\n📈 Summary Statistics:")
        for col in metrics.columns:
            if col not in ['block_id', 'trial_number']:
                values = metrics[col].dropna()
                if len(values) > 0:
                    if pd.api.types.is_timedelta64_dtype(values):
                        values_sec = values.dt.total_seconds()
                        print(f"   {col}: {values_sec.mean():.2f} ± {values_sec.std():.2f} seconds (n={len(values)})")
                    elif pd.api.types.is_numeric_dtype(values):
                        print(f"   {col}: {values.mean():.2f} ± {values.std():.2f} (n={len(values)})")
        
        print("\n" + "="*60)
        print("STEP 5: TRAJECTORY EXTRACTION")
        print("="*60)
        
        # Try to extract trajectories for detected bodyparts
        trajectories = block.extract_bodypart_trajectories()
        
        if not trajectories:
            print("⚠️  No trajectories with auto-detection, trying specific bodyparts...")
            # Try common bodypart names
            for bodyparts in [['nose', 'backOfHead'], ['nose'], ['head'], ['body']]:
                trajectories = block.extract_bodypart_trajectories(bodyparts)
                if trajectories:
                    print(f"✅ Found trajectories with bodyparts: {bodyparts}")
                    break
        
        if trajectories:
            print(f"\n🎯 Trajectory Extraction Results:")
            for bodypart, traj_df in trajectories.items():
                trials_with_data = traj_df['trial_number'].nunique()
                total_points = len(traj_df)
                print(f"   {bodypart}: {total_points} points across {trials_with_data} trials")
                
                # Show sample trajectory data
                print(f"   Sample data for {bodypart}:")
                print(traj_df.head(3))
        else:
            print("⚠️  No trajectories extracted - check bodypart names")
        
        print("\n" + "="*60)
        print("STEP 6: VISUALIZATION GENERATION")
        print("="*60)
        
        # Create all visualizations
        if trajectories:
            print(f"🎨 Creating trajectory plots...")
            for bodypart in trajectories.keys():
                block.create_trial_plots(bodypart, save_individual=True, save_combined=True)
                print(f"   ✅ Created plots for {bodypart}")
                break  # Just do the first bodypart for this test
        
        print(f"📊 Creating metrics plots...")
        block.create_metrics_plots()
        
        print("\n" + "="*60)
        print("STEP 7: COMPREHENSIVE SUMMARY")
        print("="*60)
        
        block.summary()
        
        # List all generated files
        print(f"\n📁 Generated Files:")
        output_files = sorted(list(output_dir.glob("*")))
        for file in output_files:
            if file.is_file():
                file_size = file.stat().st_size / 1024  # KB
                print(f"   📄 {file.name} ({file_size:.1f} KB)")
        
        print(f"\n✅ Full day analysis complete!")
        print(f"📁 All outputs saved to: {output_dir}")
        
        return block
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def analyze_specific_trials(block: CheeseboardBlock):
    """Detailed analysis of specific trials"""
    
    if block.trial_metrics is None:
        print("⚠️  No trial metrics available")
        return
    
    print("\n" + "="*60)
    print("DETAILED TRIAL ANALYSIS")
    print("="*60)
    
    # Find trials with different performance characteristics
    metrics = block.trial_metrics
    
    # Trial with fastest first reward
    if 'first_reward_time' in metrics.columns:
        fastest_trial = metrics.loc[metrics['first_reward_time'].idxmin()]
        print(f"\n🏃 Fastest First Reward:")
        print(f"   Trial {fastest_trial['trial_number']}: {fastest_trial['first_reward_time'].total_seconds():.2f} seconds")
    
    # Trial with most rewards
    if 'num_rewards_collected' in metrics.columns:
        best_trial = metrics.loc[metrics['num_rewards_collected'].idxmax()]
        print(f"\n🏆 Most Rewards Collected:")
        print(f"   Trial {best_trial['trial_number']}: {best_trial['num_rewards_collected']} rewards")
    
    # Trial with longest duration
    if 'trial_length' in metrics.columns:
        longest_trial = metrics.loc[metrics['trial_length'].idxmax()]
        print(f"\n⏱️  Longest Trial:")
        print(f"   Trial {longest_trial['trial_number']}: {longest_trial['trial_length'].total_seconds():.2f} seconds")
    
    # Learning trend analysis
    if len(metrics) > 1 and 'first_reward_time' in metrics.columns:
        first_half = metrics.iloc[:len(metrics)//2]['first_reward_time'].dropna()
        second_half = metrics.iloc[len(metrics)//2:]['first_reward_time'].dropna()
        
        if len(first_half) > 0 and len(second_half) > 0:
            first_mean = first_half.dt.total_seconds().mean()
            second_mean = second_half.dt.total_seconds().mean()
            improvement = first_mean - second_mean
            
            print(f"\n📈 Learning Analysis:")
            print(f"   First half avg: {first_mean:.2f} seconds")
            print(f"   Second half avg: {second_mean:.2f} seconds")
            print(f"   Improvement: {improvement:.2f} seconds {'✅' if improvement > 0 else '➡️'}")

if __name__ == "__main__":
    print("🧀 Comprehensive Single Day Analysis")
    print("="*50)
    
    # Run the full analysis
    block = test_full_day_analysis()
    
    if block is not None:
        # Additional detailed analysis
        analyze_specific_trials(block)
        
        print(f"\n🎉 Analysis Complete!")
        print(f"📊 Check all plots and data in: /Users/nick/Projects/cheeseboardAnalysis/DATA/FULL_DAY_TEST")
    else:
        print(f"\n❌ Analysis failed - check error messages above")
