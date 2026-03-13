"""
Keep only runs that have actual logs (logs/run.log), delete the rest, and rename
kept runs to run_001, run_002, ... (newest first).
"""
import json
import shutil
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.json_utils import json_safe

PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "outputs" / "runs"


def has_actual_log(run_dir: Path) -> bool:
    """True if this run has a non-empty run.log."""
    log_file = run_dir / "logs" / "run.log"
    if not log_file.exists():
        return False
    return log_file.stat().st_size > 0


def update_metadata_run_id(run_dir: Path, new_run_id: str) -> None:
    """Update run_id in all run_metadata.json files under this run."""
    for path in [run_dir / "run_metadata.json", run_dir / "reports" / "run_metadata.json"]:
        if path.exists():
            try:
                with open(path, "r") as f:
                    meta = json.load(f)
                meta["run_id"] = new_run_id
                with open(path, "w") as f:
                    json.dump(meta, f, indent=2, default=json_safe)
            except Exception as e:
                print(f"  Warning: could not update {path}: {e}")


def main():
    runs_dir = RUNS_DIR
    if not runs_dir.exists():
        print("No outputs/runs directory found.")
        return

    # Find all run directories
    all_runs = [d for d in runs_dir.iterdir() if d.is_dir()]
    # Keep only those with actual logs
    runs_with_logs = [d for d in all_runs if has_actual_log(d)]
    to_delete = [d for d in all_runs if d not in runs_with_logs]

    print(f"Total run directories: {len(all_runs)}")
    print(f"Runs with actual logs (keeping): {len(runs_with_logs)}")
    print(f"Runs without logs (deleting): {len(to_delete)}")

    # Sort kept runs by modification time, newest first
    runs_with_logs.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    # Delete runs without logs
    for run_dir in to_delete:
        print(f"  Deleting: {run_dir.name}")
        shutil.rmtree(run_dir, ignore_errors=True)

    # Rename kept runs to run_001, run_002, ...
    for i, run_dir in enumerate(runs_with_logs, start=1):
        new_name = f"run_{i:03d}"
        new_path = run_dir.parent / new_name
        if run_dir.name == new_name:
            continue
        # If target already exists (e.g. from previous run of this script), remove it
        if new_path.exists():
            shutil.rmtree(new_path, ignore_errors=True)
        run_dir.rename(new_path)
        update_metadata_run_id(new_path, new_name)
        print(f"  Renamed: {run_dir.name} -> {new_name}")

    print("Done.")


if __name__ == "__main__":
    main()
