#!/usr/bin/env python3
"""
Simple script to analyze a specific experiment day
Just modify the block_id variable below to choose your experiment
"""

from pathlib import Path
from single_block_processor import CheeseboardBlock

# 🎯 CHOOSE YOUR EXPERIMENT HERE:
# Available options:
# "ExperimentVideo_2025-08-12_1407" - Aug 12, 2:07 PM (77k frames)
# "ExperimentVideo_2025-08-12_1452" - Aug 12, 2:52 PM (67k frames)
# "ExperimentVideo_2025-08-13_1105" - Aug 13, 11:05 AM (60k frames)
# "ExperimentVideo_2025-08-13_1136" - Aug 13, 11:36 AM (65k frames)
# "ExperimentVideo_2025-08-14_1245" - Aug 14, 12:45 PM (195k frames) ⭐ LONGEST
# "ExperimentVideo_2025-08-14_1426" - Aug 14, 2:26 PM (201k frames) ⭐ LONGEST
# "ExperimentVideo_2025-08-15_1042" - Aug 15, 10:42 AM (54k frames)
# "ExperimentVideo_2025-08-15_1110" - Aug 15, 11:10 AM (56k frames)
# "ExperimentVideo_2025-08-15_1146" - Aug 15, 11:46 AM (42k frames)
# "ExperimentVideo_2025-08-15_1207" - Aug 15, 12:07 PM (37k frames)

CHOSEN_EXPERIMENT = "ExperimentVideo_2025-08-15_1110"  # 🔄 August 15, 11:10 AM session

def analyze_experiment(block_id: str):
    """Analyze the specified experiment"""
    
    print(f"🧀 Analyzing Experiment: {block_id}")
    print("="*60)
    
    # Set up paths
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA")
    output_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/ANALYSIS_OUTPUT")
    
    timestamps_file = data_dir / f"{block_id}_timestamps.csv"
    dlc_file = data_dir / f"{block_id}DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
    
    print(f"📂 Data files:")
    print(f"   📄 Timestamps: {timestamps_file}")
    print(f"   📄 DLC: {dlc_file}")
    print(f"   📁 Output: {output_dir / block_id}")
    
    # Check files exist
    if not timestamps_file.exists():
        print(f"❌ Timestamps file not found!")
        return None
    
    if not dlc_file.exists():
        print(f"❌ DLC file not found!")
        return None
    
    print("✅ All files found!")
    
    try:
        # Create processor
        block = CheeseboardBlock(
            block_id=block_id,
            timestamps_file=str(timestamps_file),
            dlc_file=str(dlc_file),
            output_dir=str(output_dir / block_id)
        )
        
        # Run complete analysis
        print(f"\n🚀 Running complete analysis pipeline...")
        result = block.process_full_block(
            bodyparts=None,  # Auto-detect bodyparts
            create_plots=True
        )
        
        if result['success']:
            print(f"\n🎉 Analysis Complete!")
            print(f"📊 Results Summary:")
            print(f"   ✅ Trials processed: {result['num_trials']}")
            print(f"   ✅ Bodyparts analyzed: {result['num_bodyparts']}")
            print(f"   ✅ Bodyparts: {result['bodyparts']}")
            print(f"   📁 Output directory: {result['output_directory']}")
            
            # Show detailed summary
            print(f"\n" + "="*60)
            print("DETAILED SUMMARY")
            print("="*60)
            block.summary()
            
            # List all generated files
            print(f"\n📁 All Generated Files:")
            output_path = Path(result['output_directory'])
            files = sorted(list(output_path.glob("*")))
            
            csv_files = [f for f in files if f.suffix == '.csv']
            png_files = [f for f in files if f.suffix == '.png']
            other_files = [f for f in files if f.suffix not in ['.csv', '.png']]
            
            if csv_files:
                print(f"   📊 Data files ({len(csv_files)}):")
                for f in csv_files:
                    size_kb = f.stat().st_size / 1024
                    print(f"      📄 {f.name} ({size_kb:.1f} KB)")
            
            if png_files:
                print(f"   🎨 Plot files ({len(png_files)}):")
                for f in png_files:
                    size_kb = f.stat().st_size / 1024
                    print(f"      🖼️  {f.name} ({size_kb:.1f} KB)")
            
            if other_files:
                print(f"   📋 Other files ({len(other_files)}):")
                for f in other_files:
                    print(f"      📄 {f.name}")
            
            print(f"\n🎯 To view your results:")
            print(f"   📁 Open folder: {output_path}")
            print(f"   🖼️  View plots: Look for .png files")
            print(f"   📊 Check data: Look at _metrics.csv and _trajectory.csv files")
            
            return block
            
        else:
            print(f"\n❌ Analysis failed!")
            print(f"Errors: {result['errors']}")
            return None
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🧀 Simple Experiment Analyzer")
    print("="*50)
    
    print(f"🎯 Chosen experiment: {CHOSEN_EXPERIMENT}")
    print(f"💡 To choose a different experiment, edit the CHOSEN_EXPERIMENT variable at the top of this file")
    
    # Run the analysis
    block = analyze_experiment(CHOSEN_EXPERIMENT)
    
    if block is not None:
        print(f"\n✅ Analysis completed successfully!")
        print(f"🎉 Check your results in the output directory!")
    else:
        print(f"\n❌ Analysis failed. Check the error messages above.")
