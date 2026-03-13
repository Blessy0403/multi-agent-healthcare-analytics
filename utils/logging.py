"""
Logging utilities for Multi-Agent Healthcare Analytics Pipeline.

Provides structured logging with agent-specific contexts and collaboration tracking.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
from contextlib import contextmanager
import time

from utils.config import get_config
from utils.json_utils import json_safe


class AgentLogger:
    """
    Custom logger for agent-specific logging with collaboration tracking.
    
    Each agent gets its own logger instance that tracks:
    - Agent execution time
    - Input/output artifacts
    - Errors and warnings
    - Handover events
    """
    
    def __init__(self, agent_name: str, log_dir: Optional[Path] = None):
        """
        Initialize agent logger.
        
        Args:
            agent_name: Name of the agent (e.g., 'data_agent', 'model_agent')
            log_dir: Directory for log files (defaults to config log_dir)
        """
        self.agent_name = agent_name
        config = get_config()
        self.log_dir = log_dir or config.log_dir
        
        # Create logger
        self.logger = logging.getLogger(f'pipeline.{agent_name}')
        self.logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_format = logging.Formatter(
                f'[%(asctime)s] [%(levelname)s] [{agent_name}] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
            
            # File handler
            log_file = self.log_dir / f'{agent_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_format = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
        
        # Collaboration tracking
        self.collaboration_log = []
        self.start_time = None
        self.end_time = None
    
    def log_artifact(self, artifact_type: str, artifact_path: str, metadata: Optional[dict] = None):
        """
        Log an artifact produced by the agent.
        
        Args:
            artifact_type: Type of artifact (e.g., 'dataset', 'model', 'explanation')
            artifact_path: Path to the artifact
            metadata: Additional metadata about the artifact
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'agent': self.agent_name,
            'artifact_type': artifact_type,
            'artifact_path': str(artifact_path),
            'metadata': metadata or {}
        }
        self.collaboration_log.append(log_entry)
        self.logger.info(f"Produced artifact: {artifact_type} at {artifact_path}")
    
    def log_handover(self, to_agent: str, artifacts: list, metadata: Optional[dict] = None):
        """
        Log a handover to another agent.
        
        Args:
            to_agent: Name of the receiving agent
            artifacts: List of artifact paths being passed
            metadata: Additional handover metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'from_agent': self.agent_name,
            'to_agent': to_agent,
            'artifacts': artifacts,
            'metadata': metadata or {}
        }
        self.collaboration_log.append(log_entry)
        self.logger.info(f"Handover to {to_agent}: {len(artifacts)} artifacts")
    
    @contextmanager
    def execution_timer(self):
        """Context manager to track agent execution time."""
        self.start_time = time.time()
        self.logger.info(f"{self.agent_name} execution started")
        try:
            yield
        finally:
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            self.logger.info(f"{self.agent_name} execution completed in {execution_time:.2f} seconds")
            
            # Log execution time
            self.collaboration_log.append({
                'timestamp': datetime.now().isoformat(),
                'agent': self.agent_name,
                'event': 'execution_complete',
                'execution_time_seconds': execution_time
            })
    
    def save_collaboration_log(self, output_path: Optional[Path] = None):
        """Save collaboration log to JSON file."""
        if output_path is None:
            output_path = self.log_dir / f'{self.agent_name}_collaboration.json'
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            return obj
        
        serializable_log = convert_paths(self.collaboration_log)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_log, f, indent=2, default=json_safe)
        
        self.logger.info(f"Collaboration log saved to {output_path}")
    
    def error(self, message: str, *args, **kwargs):
        """Log an error. Accepts *args for printf-style (message % args). Ignores exc_info and other kwargs."""
        msg = message % args if args else message
        self.logger.error(msg)
    
    def warning(self, message: str, *args, **kwargs):
        """Log a warning. Accepts *args for printf-style. Ignores kwargs."""
        msg = message % args if args else message
        self.logger.warning(msg)
    
    def info(self, message: str, *args, **kwargs):
        """Log an info message. Accepts *args for printf-style. Ignores kwargs."""
        msg = message % args if args else message
        self.logger.info(msg)
    
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message. Accepts *args for printf-style. Ignores kwargs."""
        msg = message % args if args else message
        self.logger.debug(msg)


def setup_root_logger(log_dir: Optional[Path] = None):
    """Setup root logger for the pipeline. Writes to console and optionally to run log file."""
    try:
        config = get_config()
        run_log_dir = log_dir or config.log_dir
    except Exception:
        run_log_dir = None
    
    root_logger = logging.getLogger('pipeline')
    root_logger.setLevel(logging.INFO)
    
    # Console handler (only if not already present)
    if not any(h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)):
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        root_logger.addHandler(console)
    
    # Per-run log file: always write pipeline output to run directory
    if run_log_dir is not None:
        run_log_dir = Path(run_log_dir)
        run_log_dir.mkdir(parents=True, exist_ok=True)
        run_log_path = run_log_dir / 'run.log'
        has_run_file = any(
            getattr(h, 'baseFilename', None) == str(run_log_path)
            for h in root_logger.handlers
            if isinstance(h, logging.FileHandler)
        )
        if not has_run_file:
            try:
                file_handler = logging.FileHandler(run_log_path, mode='a', encoding='utf-8')
                file_handler.setLevel(logging.DEBUG)
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                ))
                root_logger.addHandler(file_handler)
            except Exception:
                pass
    
    return root_logger
