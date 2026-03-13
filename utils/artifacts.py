"""Artifact management for agent outputs."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np

from utils.json_utils import json_safe


class ArtifactManager:
    """Manages saving and loading of pipeline artifacts."""
    
    def __init__(self, run_id: str):
        """Initialize with a run ID."""
        self.run_id = run_id
        self.run_dir = Path(__file__).parent.parent / 'outputs' / 'runs' / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
    
    def save_json(self, data: Any, filename: str, subdir: str = 'reports') -> Path:
        """Save data as JSON."""
        target_dir = self.run_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = target_dir / filename
        
        # Convert numpy types to native Python
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        serializable_data = convert_numpy(data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2, default=json_safe)
        
        return filepath
    
    def load_json(self, filename: str, subdir: str = 'reports') -> Any:
        """Load data from JSON."""
        filepath = self.run_dir / subdir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found: {filepath}")
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def save_pickle(self, data: Any, filename: str, subdir: str = 'models') -> Path:
        """Save data as pickle."""
        target_dir = self.run_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = target_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        return filepath
    
    def load_pickle(self, filename: str, subdir: str = 'models') -> Any:
        """Load data from pickle."""
        filepath = self.run_dir / subdir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def save_csv(self, df: pd.DataFrame, filename: str, subdir: str = 'reports') -> Path:
        """Save DataFrame as CSV."""
        target_dir = self.run_dir / subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = target_dir / filename
        df.to_csv(filepath, index=False)
        
        return filepath
    
    def load_csv(self, filename: str, subdir: str = 'reports') -> pd.DataFrame:
        """Load DataFrame from CSV."""
        filepath = self.run_dir / subdir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Artifact not found: {filepath}")
        
        return pd.read_csv(filepath)
    
    def save_metadata(self, metadata: Dict[str, Any]) -> Path:
        """Save run metadata to both reports directory and root."""
        # Save to reports directory (primary location)
        reports_path = self.save_json(metadata, 'run_metadata.json', 'reports')
        
        # Also save to root for easier access
        root_path = self.run_dir / 'run_metadata.json'
        with open(root_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=json_safe)
        
        return reports_path
    
    def get_artifact_path(self, filename: str, subdir: str = 'reports') -> Path:
        """Get path to an artifact without loading it."""
        return self.run_dir / subdir / filename
