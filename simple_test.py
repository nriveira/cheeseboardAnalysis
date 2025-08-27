#!/usr/bin/env python3
"""
Simple test to verify the DLC header processing works
"""

import pandas as pd

def test_dlc_header_processing():
    """Test the DLC header combining logic"""
    
    print("🔬 Testing DLC 3-row header processing...")
    
    # Read the sample DLC file as raw data
    raw_df = pd.read_csv('sample_dlc_filtered.csv', header=None)
    print(f"Raw DLC file shape: {raw_df.shape}")
    print("Raw DLC content:")
    print(raw_df)
    print()
    
    # Extract the 3 header rows
    model_row = raw_df.iloc[0].fillna('')  # Row 0: model info
    bodypart_row = raw_df.iloc[1].fillna('')  # Row 1: bodypart names
    coord_row = raw_df.iloc[2].fillna('')  # Row 2: x/y/likelihood
    
    print("Header rows:")
    print(f"Model row: {list(model_row)}")
    print(f"Bodypart row: {list(bodypart_row)}")
    print(f"Coordinate row: {list(coord_row)}")
    print()
    
    # Create combined column names
    combined_columns = []
    for i in range(len(bodypart_row)):
        model = str(model_row.iloc[i]).strip()
        bodypart = str(bodypart_row.iloc[i]).strip()
        coord = str(coord_row.iloc[i]).strip()
        
        # Skip empty columns
        if not bodypart or bodypart == '' or bodypart.lower() == 'nan':
            combined_columns.append(f"empty_col_{i}")
            continue
        
        # Create meaningful column name
        if coord.lower() in ['x', 'y', 'likelihood']:
            combined_name = f"{bodypart}_{coord}"
        else:
            combined_name = f"{bodypart}_{coord}" if coord else bodypart
        
        combined_columns.append(combined_name)
    
    print(f"Combined column names: {combined_columns}")
    print()
    
    # Get the actual data (skip the 3 header rows)
    data_df = raw_df.iloc[3:].reset_index(drop=True)
    data_df.columns = combined_columns
    
    # Convert numeric columns
    for col in data_df.columns:
        if '_x' in col or '_y' in col or '_likelihood' in col:
            data_df[col] = pd.to_numeric(data_df[col], errors='coerce')
    
    print("Final processed DLC data:")
    print(data_df)
    print(f"Data types: {data_df.dtypes}")
    print()
    
    # Test bodypart detection
    bodyparts = set()
    for col in data_df.columns:
        if '_x' in col or '_y' in col or '_likelihood' in col:
            # Extract bodypart name (everything before the last underscore)
            parts = col.split('_')
            if len(parts) >= 2:
                # Get everything except the last part (which should be x, y, or likelihood)
                bodypart = '_'.join(parts[:-1])
                if bodypart and not bodypart.startswith('empty_col'):
                    bodyparts.add(bodypart)
    
    detected_bodyparts = sorted(list(bodyparts))
    print(f"✅ Detected bodyparts: {detected_bodyparts}")
    
    return True

if __name__ == "__main__":
    test_dlc_header_processing()
