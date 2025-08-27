#!/usr/bin/env python3
"""
Test script for the updated DLC header processing
"""

from single_block_processor import CheeseboardBlock
import os

def main():
    # Test with sample files
    try:
        processor = CheeseboardBlock('sample', 'sample_timestamps.csv', 'sample_dlc_filtered.csv')
        
        print('🔬 Testing DLC header processing...')
        print()
        
        # Test data loading
        success = processor.load_data()
        if success:
            print('✅ Data loading successful!')
            print(f'   Raw DLC shape: {processor.raw_dlc.shape}')
            print(f'   DLC columns: {list(processor.raw_dlc.columns)}')
            print(f'   First few rows:')
            print(processor.raw_dlc.head())
            print()
            
            # Test integration  
            success = processor.integrate_data()
            if success:
                print('✅ Data integration successful!')
                print(f'   Combined data columns: {list(processor.combined_data.columns)}')
                
                # Test bodypart detection
                bodyparts = processor._detect_bodyparts()
                print(f'   Detected bodyparts: {bodyparts}')
                
                return True
            else:
                print('❌ Integration failed')
                return False
        else:
            print('❌ Loading failed')
            return False
    
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()
