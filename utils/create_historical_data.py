"""Create historical raw data files with past dates."""

import shutil
from pathlib import Path
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / 'data' / 'raw'


def create_historical_files(num_files: int = 18):
    """Create historical raw data files with past dates."""
    
    # Source files
    heart_source = RAW_DATA_DIR / 'heart_disease_raw_data_20260120.data'
    diabetes_source = RAW_DATA_DIR / 'diabetes_raw_data_20260120.data'
    
    if not heart_source.exists():
        print(f"Source file not found: {heart_source}")
        return
    
    if not diabetes_source.exists():
        print(f"Source file not found: {diabetes_source}")
        return
    
    print(f"Creating {num_files} historical data files...")
    
    # Create files going back in time
    base_date = datetime(2026, 1, 20)
    
    for i in range(num_files):
        # Go back in time (every 3-4 days)
        days_back = i * 3
        file_date = base_date - timedelta(days=days_back)
        date_str = file_date.strftime("%Y%m%d")
        
        # Heart disease file
        heart_dest = RAW_DATA_DIR / f'heart_disease_raw_data_{date_str}.data'
        if not heart_dest.exists():
            shutil.copy2(heart_source, heart_dest)
            print(f"Created: {heart_dest.name}")
        
        # Diabetes file
        diabetes_dest = RAW_DATA_DIR / f'diabetes_raw_data_{date_str}.data'
        if not diabetes_dest.exists():
            shutil.copy2(diabetes_source, diabetes_dest)
            print(f"Created: {diabetes_dest.name}")
    
    print(f"\n✅ Created historical data files!")
    print(f"Total files in data/raw/: {len(list(RAW_DATA_DIR.glob('*_raw_data_*.data')))}")


if __name__ == '__main__':
    create_historical_files(18)
