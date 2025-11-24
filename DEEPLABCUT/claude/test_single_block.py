#!/usr/bin/env python3
"""
Test script for the single_block_processor.py
Demonstrates how to use the new CheeseboardBlock class with your actual data.
"""

import os
from pathlib import Path
from single_block_processor import CheeseboardBlock, process_single_block

def test_single_block_processor():
    """Test the single block processor with actual data"""
    
    # Set up paths for your data
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA")
    output_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/SINGLE_BLOCK_OUTPUT")
    
    # Pick a specific block to test
    block_id = "ExperimentVideo_2025-08-13_1105"
    timestamps_file = data_dir / f"{block_id}_timestamps.csv"
    dlc_file = data_dir / f"{block_id}DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
    
    print(f"🧪 Testing single block processor with {block_id}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Timestamps file: {timestamps_file}")
    print(f"📄 DLC file: {dlc_file}")
    
    # Check if files exist
    if not timestamps_file.exists():
        print(f"❌ Timestamps file not found: {timestamps_file}")
        return
    
    if not dlc_file.exists():
        print(f"❌ DLC file not found: {dlc_file}")
        return
    
    print("✅ All files found, proceeding with processing...")
    
    try:
        # Method 1: Using the convenience function
        print("\n" + "="*60)
        print("METHOD 1: Using process_single_block() convenience function")
        print("="*60)
        
        block = process_single_block(
            block_id=block_id,
            timestamps_file=str(timestamps_file),
            dlc_file=str(dlc_file),
            output_dir=str(output_dir),
            bodyparts=None  # Auto-detect bodyparts
        )
        
        # Print summary
        block.summary()
        
        # Method 2: Using the class directly (more control)
        print("\n" + "="*60)
        print("METHOD 2: Using CheeseboardBlock class directly")
        print("="*60)
        
        # Create a different output directory for comparison
        output_dir_2 = output_dir / "method2_test"
        
        block2 = CheeseboardBlock(
            block_id=f"{block_id}_method2",
            timestamps_file=str(timestamps_file),
            dlc_file=str(dlc_file),
            output_dir=str(output_dir_2)
        )
        
        # Process step by step with more control
        print("🔄 Step 1: Loading data...")
        if not block2.load_data():
            print("❌ Failed to load data")
            return
        
        print("🔄 Step 2: Inspecting DLC structure...")
        block2.inspect_dlc_structure()
        
        print("🔄 Step 3: Integrating data...")
        integrated = block2.integrate_data()
        if integrated is None:
            print("❌ Failed to integrate data")
            return
        
        print("🔄 Step 4: Segmenting trials...")
        segments = block2.segment_trials()
        if segments is None:
            print("❌ Failed to segment trials")
            return
        
        print(f"✅ Found {len(segments)} trials")
        print("Trial segments preview:")
        print(segments.head())
        
        print("🔄 Step 5: Computing metrics...")
        metrics = block2.compute_trial_metrics()
        if metrics is None:
            print("❌ Failed to compute metrics")
            return
        
        print("Metrics preview:")
        print(metrics.head())
        
        print("🔄 Step 6: Extracting trajectories...")
        # Try specific bodyparts first
        trajectories = block2.extract_bodypart_trajectories(['nose', 'backOfHead'])
        
        if not trajectories:
            print("⚠️  No trajectories extracted with specified bodyparts, trying auto-detection...")
            trajectories = block2.extract_bodypart_trajectories(None)
        
        print(f"✅ Extracted trajectories for: {list(trajectories.keys())}")
        
        print("🔄 Step 7: Creating plots...")
        if trajectories:
            block2.create_trial_plots(list(trajectories.keys())[0])
        block2.create_metrics_plots()
        
        print("🔄 Step 8: Summary...")
        block2.summary()
        
        print("\n✅ Testing complete!")
        print(f"📁 Check outputs in: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

def batch_test_multiple_blocks():
    """Test processing multiple blocks"""
    
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA")
    output_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/BATCH_OUTPUT")
    
    # Find all available blocks
    timestamp_files = list(data_dir.glob("*_timestamps.csv"))
    
    print(f"🔍 Found {len(timestamp_files)} timestamp files")
    
    for i, ts_file in enumerate(timestamp_files[:3]):  # Process first 3 blocks only
        # Extract block ID from filename
        block_id = ts_file.stem.replace("_timestamps", "")
        dlc_file = data_dir / f"{block_id}DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
        
        if not dlc_file.exists():
            print(f"⚠️  Skipping {block_id} - DLC file not found")
            continue
        
        print(f"\n📊 Processing block {i+1}/3: {block_id}")
        
        try:
            block = process_single_block(
                block_id=block_id,
                timestamps_file=str(ts_file),
                dlc_file=str(dlc_file),
                output_dir=str(output_dir)
            )
            print(f"✅ Successfully processed {block_id}")
            
        except Exception as e:
            print(f"❌ Failed to process {block_id}: {e}")

if __name__ == "__main__":
    print("🧀 Testing Single Block Processor")
    print("="*50)
    
    # Test single block first
    test_single_block_processor()
    
    # Uncomment to test batch processing
    # print("\n" + "="*50)
    # print("🔄 Testing batch processing...")
    # batch_test_multiple_blocks()
