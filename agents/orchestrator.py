"""
Orchestrator: Backward-compatible wrapper around the central pipeline orchestrator.

The canonical pipeline is in pipeline/orchestrator.py:
  DataAgent → ModelAgent → ExplainabilityAgent → EvaluationAgent → FeedbackAgent → CrossDatasetAgent (if enabled)
"""

from typing import Dict, Any, Optional

from utils.config import PipelineConfig, get_config
from utils.logging import AgentLogger


class Orchestrator:
    """
    Wrapper that delegates to pipeline.orchestrator.execute_pipeline().
    Kept for backward compatibility (e.g. Orchestrator(config).execute_pipeline()).
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or get_config()
        self.logger = AgentLogger("orchestrator")

    def execute_pipeline(self) -> Dict[str, Any]:
        """Run the full pipeline via pipeline.orchestrator."""
        from pipeline.orchestrator import execute_pipeline
        return execute_pipeline(self.config)
