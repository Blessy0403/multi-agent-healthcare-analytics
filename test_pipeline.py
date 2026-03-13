"""Quick smoke test for pipeline components."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.config import get_config
from utils.seed import set_seed
from utils.logging import setup_root_logger

def test_imports():
    """Test all imports."""
    print("Testing imports...")
    try:
        from agents.data_agent import DataAgent
        from agents.model_agent import ModelAgent
        from agents.explainability_agent import ExplainabilityAgent
        from agents.orchestrator import Orchestrator
        from baseline.single_model_pipeline import SingleModelPipeline
        from evaluation.metrics import MetricsEvaluator
        from evaluation.explainability_eval import ExplainabilityEvaluator
        from evaluation.collaboration_eval import CollaborationEvaluator
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test configuration."""
    print("\nTesting configuration...")
    try:
        set_seed(42)
        config = get_config()
        print(f"✅ Config loaded: Run ID = {config.run_id}")
        print(f"✅ Run directory: {config.run_dir}")
        print(f"✅ Datasets: {config.data.datasets}")
        return True
    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_agents_init():
    """Test agent initialization."""
    print("\nTesting agent initialization...")
    try:
        config = get_config()
        from agents.data_agent import DataAgent
        from agents.model_agent import ModelAgent
        from agents.explainability_agent import ExplainabilityAgent
        
        data_agent = DataAgent(config.data)
        model_agent = ModelAgent(config.model)
        explain_agent = ExplainabilityAgent(config.explainability)
        
        print("✅ All agents initialized")
        return True
    except Exception as e:
        print(f"❌ Agent init error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dashboard():
    """Test dashboard components."""
    print("\nTesting dashboard...")
    try:
        from dashboard.components.layout import get_available_runs
        from dashboard.components.charts import plot_roc_curve
        runs = get_available_runs()
        print(f"✅ Dashboard components OK (found {len(runs)} runs)")
        return True
    except Exception as e:
        print(f"❌ Dashboard error: {e}")
        return False

if __name__ == '__main__':
    print("="*60)
    print("Pipeline Smoke Test")
    print("="*60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Agents", test_agents_init()))
    results.append(("Dashboard", test_dashboard()))
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print("\n" + "="*60)
    if all_passed:
        print("✅ All tests passed! Pipeline is ready to run.")
        print("\nNext step: Run 'python main.py' to execute the full pipeline")
    else:
        print("❌ Some tests failed. Please fix errors before running pipeline.")
    print("="*60)
