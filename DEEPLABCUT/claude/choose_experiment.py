#!/usr/bin/env python3
"""
Interactive script to choose and analyze a specific experimental day
"""

import pandas as pd
from pathlib import Path
from single_block_processor import CheeseboardBlock

def show_available_experiments():
    """Display all available experiments with details"""
    
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA")
    timestamp_files = sorted(list(data_dir.glob("*_timestamps.csv")))
    
    print("📅 Available Experimental Sessions:")
    print("="*60)
    
    experiments = []
    for i, ts_file in enumerate(timestamp_files):
        block_id = ts_file.stem.replace("_timestamps", "")
        dlc_file = data_dir / f"{block_id}DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
        
        # Get file sizes
        ts_size = ts_file.stat().st_size / (1024 * 1024)  # MB
        dlc_exists = dlc_file.exists()
        
        # Extract date and time
        parts = block_id.split("_")
        date = parts[1] if len(parts) > 1 else "unknown"
        time = parts[2] if len(parts) > 2 else "unknown"
        
        experiments.append({
            'index': i + 1,
            'block_id': block_id,
            'date': date,
            'time': time,
            'ts_file': ts_file,
            'dlc_file': dlc_file,
            'ts_size_mb': ts_size,
            'dlc_exists': dlc_exists,
            'estimated_frames': int(ts_size * 1024 * 34)  # Rough estimate
        })
        
        status = "✅" if dlc_exists else "❌"
        print(f"{i+1:2d}. {block_id}")
        print(f"    📅 Date: {date}, ⏰ Time: {time}")
        print(f"    📊 Size: {ts_size:.1f} MB (~{int(ts_size * 1024 * 34):,} frames)")
        print(f"    🎯 DLC file: {status}")
        print()
    
    return experiments

def analyze_chosen_experiment(block_id: str):
    """Analyze the chosen experiment"""
    
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA")
    output_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/CHOSEN_ANALYSIS")
    
    timestamps_file = data_dir / f"{block_id}_timestamps.csv"
    dlc_file = data_dir / f"{block_id}DLC_resnet50_liveTestAug8shuffle1_100000_filtered.csv"
    
    print(f"\n🚀 Starting analysis of: {block_id}")
    print("="*60)
    
    if not timestamps_file.exists():
        print(f"❌ Timestamps file not found: {timestamps_file}")
        return None
    
    if not dlc_file.exists():
        print(f"❌ DLC file not found: {dlc_file}")
        return None
    
    # Create and run the analysis
    block = CheeseboardBlock(
        block_id=block_id,
        timestamps_file=str(timestamps_file),
        dlc_file=str(dlc_file),
        output_dir=str(output_dir / block_id)
    )
    
    print("🔄 Running complete analysis pipeline...")
    result = block.process_full_block(create_plots=True)
    
    if result['success']:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Results:")
        print(f"   - Trials found: {result['num_trials']}")
        print(f"   - Bodyparts analyzed: {result['num_bodyparts']}")
        print(f"   - Bodyparts: {result['bodyparts']}")
        print(f"   - Output directory: {result['output_directory']}")
        
        # Show summary
        block.summary()
        
        # List generated files
        print(f"\n📁 Generated Files:")
        output_files = sorted(list(Path(result['output_directory']).glob("*")))
        for file in output_files:
            if file.is_file():
                file_size = file.stat().st_size / 1024  # KB
                print(f"   📄 {file.name} ({file_size:.1f} KB)")
        
        return block
    else:
        print(f"❌ Analysis failed: {result['errors']}")
        return None

def quick_preview_experiment(block_id: str):
    """Quick preview of experiment without full processing"""
    
    data_dir = Path("/Users/nick/Projects/cheeseboardAnalysis/DATA/RECORDED DATA")
    timestamps_file = data_dir / f"{block_id}_timestamps.csv"
    
    if not timestamps_file.exists():
        print(f"❌ File not found: {timestamps_file}")
        return
    
    # Quick peek at the data
    print(f"\n👀 Quick Preview of {block_id}:")
    print("-" * 40)
    
    # Load timestamps
    timestamps = pd.read_csv(timestamps_file, header=None, names=['UnixTime', 'Monotonic', 'Event'])
    
    # Basic stats
    print(f"📊 Basic Stats:")
    print(f"   Total frames: {len(timestamps):,}")
    
    # Event analysis
    events = timestamps['Event'].value_counts().sort_index()
    print(f"   Event counts: {dict(events)}")
    
    # Estimate trial count (count of event 1s)
    trial_count = events.get(1, 0)
    print(f"   Estimated trials: {trial_count}")
    
    # Duration estimate
    if len(timestamps) > 1:
        start_time = pd.to_datetime(timestamps['UnixTime'].iloc[0], unit='s')
        end_time = pd.to_datetime(timestamps['UnixTime'].iloc[-1], unit='s')
        duration = end_time - start_time
        print(f"   Duration: {duration}")
    
    print()

if __name__ == "__main__":
    print("🧀 Experimental Day Selector")
    print("="*50)
    
    # Show all available experiments
    experiments = show_available_experiments()
    
    print("Options:")
    print("1. Enter a number (1-10) to analyze that experiment")
    print("2. Enter 'preview X' to quickly preview experiment X")
    print("3. Enter a specific block ID (e.g., 'ExperimentVideo_2025-08-14_1245')")
    print("4. Enter 'all' to see suggestions based on data size")
    
    choice = input("\n🎯 Your choice: ").strip()
    
    if choice.lower() == 'all':
        print("\n💡 Recommendations:")
        print("🔸 For longest session with most data: ExperimentVideo_2025-08-14_1426")
        print("🔸 For medium-length session: ExperimentVideo_2025-08-13_1105")
        print("🔸 For quick test: ExperimentVideo_2025-08-15_1207")
        
    elif choice.startswith('preview '):
        try:
            idx = int(choice.split()[1]) - 1
            if 0 <= idx < len(experiments):
                quick_preview_experiment(experiments[idx]['block_id'])
            else:
                print("❌ Invalid experiment number")
        except:
            print("❌ Invalid preview command")
            
    elif choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(experiments):
            exp = experiments[idx]
            if exp['dlc_exists']:
                analyze_chosen_experiment(exp['block_id'])
            else:
                print(f"❌ DLC file missing for {exp['block_id']}")
        else:
            print("❌ Invalid experiment number")
            
    elif choice.startswith('ExperimentVideo_'):
        analyze_chosen_experiment(choice)
        
    else:
        print("❌ Invalid choice")
