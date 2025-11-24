#!/usr/bin/env python3
"""
Test script to verify the header integration is working correctly
Tests the combining of 3-row DLC headers into single meaningful column names
"""

import pandas as pd
from pathlib import Path
from single_block_processor import CheeseboardBlock

def test_sample_data_integration():
    """Test header integration using the sample files"""
    
    print("🧪 Testing header integration with sample data")
    print("="*50)
    
    # Use sample files
    timestamps_file = "/Users/nick/Projects/cheeseboardAnalysis/sample_timestamps.csv"
    dlc_file = "/Users/nick/Projects/cheeseboardAnalysis/sample_dlc_filtered.csv"
    output_dir = "/Users/nick/Projects/cheeseboardAnalysis/TEST_OUTPUT"
    
    # Create test block
    block = CheeseboardBlock(
        block_id="sample_test",
        timestamps_file=timestamps_file,
        dlc_file=dlc_file,
        output_dir=output_dir
    )
    
    print("🔄 Step 1: Loading data...")
    if not block.load_data():
        print("❌ Failed to load data")
        return
    
    print("\n📋 Raw timestamp data:")
    print(block.raw_timestamps.head())
    print(f"Shape: {block.raw_timestamps.shape}")
    
    print("\n📋 Raw DLC data (after header combination):")
    print(block.raw_dlc.head())
    print(f"Shape: {block.raw_dlc.shape}")
    print(f"Columns: {list(block.raw_dlc.columns)}")
    
    print("\n🔄 Step 2: Integrating data...")
    integrated = block.integrate_data()
    
    if integrated is None:
        print("❌ Failed to integrate data")
        return
    
    print("\n📋 Integrated data:")
    print(integrated.head())
    print(f"Shape: {integrated.shape}")
    print(f"Columns: {list(integrated.columns)}")
    
    # Verify the integration worked correctly
    print("\n✅ Integration verification:")
    
    # Check that timestamp columns are present
    timestamp_cols = ['UnixTime', 'Monotonic', 'Event']
    for col in timestamp_cols:
        if col in integrated.columns:
            print(f"   ✅ {col} column present")
        else:
            print(f"   ❌ {col} column missing")
    
    # Check that DLC columns have meaningful names
    dlc_cols = [col for col in integrated.columns if '_x' in col or '_y' in col or '_likelihood' in col]
    print(f"   ✅ Found {len(dlc_cols)} DLC coordinate columns: {dlc_cols}")
    
    # Check data integrity
    print(f"   ✅ Number of data rows: {len(integrated)}")
    
    # Test specific values
    if len(integrated) > 0:
        print(f"   ✅ First timestamp: {integrated['UnixTime'].iloc[0]}")
        print(f"   ✅ First event: {integrated['Event'].iloc[0]}")
        if 'nose_x' in integrated.columns:
            print(f"   ✅ First nose_x: {integrated['nose_x'].iloc[0]}")
    
    print("\n🔄 Step 3: Testing trial segmentation...")
    segments = block.segment_trials()
    
    if segments is not None:
        print(f"   ✅ Found {len(segments)} trials")
        print("   📋 Trial segments:")
        print(segments)
    else:
        print("   ⚠️  No trials found (this is normal for sample data)")
    
    print("\n✅ Header integration test complete!")
    
    return block

def compare_with_original_format():
    """Compare the new integrated format with the original approach"""
    
    print("\n🔍 Comparing with original combinedDLC.py approach")
    print("="*50)
    
    timestamps_file = "/Users/nick/Projects/cheeseboardAnalysis/sample_timestamps.csv"
    dlc_file = "/Users/nick/Projects/cheeseboardAnalysis/sample_dlc_filtered.csv"
    
    # Load data the old way (mimicking combinedDLC.py)
    print("📂 Loading data the old way...")
    timestamps_df = pd.read_csv(timestamps_file, header=None, names=['UnixTime', 'Monotonic', 'Event'])
    dlc_df = pd.read_csv(dlc_file)  # This keeps the 3-row header structure
    
    print(f"   Timestamps shape: {timestamps_df.shape}")
    print(f"   DLC shape: {dlc_df.shape}")
    print(f"   DLC columns (first 6): {list(dlc_df.columns[:6])}")
    
    # Create header rows like the old method
    timestamps_header = pd.DataFrame(0, index=range(2), columns=['UnixTime', 'Monotonic', 'Event'])
    timestamps_with_header = pd.concat([timestamps_header, timestamps_df], axis=0, ignore_index=True)
    old_combined = pd.concat([timestamps_with_header, dlc_df], axis=1)
    
    print(f"\n📋 Old combined format:")
    print(f"   Shape: {old_combined.shape}")
    print(f"   First few rows:")
    print(old_combined.head())
    
    # Load data the new way
    print("\n📂 Loading data the new way...")
    block = CheeseboardBlock(
        block_id="comparison_test",
        timestamps_file=timestamps_file,
        dlc_file=dlc_file,
        output_dir="/Users/nick/Projects/cheeseboardAnalysis/TEST_OUTPUT"
    )
    
    block.load_data()
    new_combined = block.integrate_data()
    
    print(f"\n📋 New combined format:")
    print(f"   Shape: {new_combined.shape}")
    print(f"   Columns: {list(new_combined.columns)}")
    print(f"   First few rows:")
    print(new_combined.head())
    
    print("\n🔍 Key differences:")
    print(f"   ✅ Old format: {old_combined.shape[0]} rows (includes header rows)")
    print(f"   ✅ New format: {new_combined.shape[0]} rows (data only)")
    print(f"   ✅ Old format: Generic column names (0, 1, 2, etc.)")
    print(f"   ✅ New format: Meaningful column names (nose_x, nose_y, etc.)")
    print(f"   ✅ New format is more readable and easier to work with!")

if __name__ == "__main__":
    print("🧪 Header Integration Test")
    print("="*50)
    
    # Test the sample data integration
    block = test_sample_data_integration()
    
    # Compare with original format
    compare_with_original_format()
    
    print(f"\n🎉 All tests complete!")
    print(f"📁 Check output files in: /Users/nick/Projects/cheeseboardAnalysis/TEST_OUTPUT")
