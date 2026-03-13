"""Path management utilities for run-based output structure."""

from pathlib import Path
from datetime import datetime
from typing import Optional
import uuid

PROJECT_ROOT = Path(__file__).parent.parent


def get_run_id() -> str:
    """Generate a unique run ID with readable timestamp format."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{unique_id}"


def get_run_dir(run_id: Optional[str] = None) -> Path:
    """Get the output directory for a specific run."""
    if run_id is None:
        run_id = get_run_id()
    
    run_dir = PROJECT_ROOT / 'outputs' / 'runs' / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_latest_run_id() -> Optional[str]:
    """Get the most recent run ID from outputs/runs."""
    runs_dir = PROJECT_ROOT / 'outputs' / 'runs'
    if not runs_dir.exists():
        return None
    
    runs = [d.name for d in runs_dir.iterdir() if d.is_dir()]
    if not runs:
        return None
    
    # Sort by timestamp (first part of run_id)
    runs.sort(reverse=True)
    return runs[0]


def get_run_subdirs(run_id: str) -> dict:
    """Get all subdirectories for a run."""
    run_dir = get_run_dir(run_id)
    
    subdirs = {
        'logs': run_dir / 'logs',
        'metrics': run_dir / 'metrics',
        'models': run_dir / 'models',
        'explainability': run_dir / 'explainability',
        'reports': run_dir / 'reports',
        'figures': run_dir / 'figures',
        'data': run_dir / 'data',
        'research_outputs': run_dir / 'research_outputs',
        'dashboard_outputs': run_dir / 'dashboard_outputs',
    }
    
    # Create all subdirectories
    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)
    
    return subdirs


def get_data_paths(dataset_name: str, run_id: Optional[str] = None) -> dict:
    """Get paths for dataset storage with timestamp."""
    if run_id is None:
        run_id = get_run_id()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    raw_dir = PROJECT_ROOT / 'data' / 'raw' / dataset_name / timestamp
    processed_dir = PROJECT_ROOT / 'data' / 'processed' / dataset_name / timestamp
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        'raw': raw_dir,
        'processed': processed_dir,
        'timestamp': timestamp
    }
