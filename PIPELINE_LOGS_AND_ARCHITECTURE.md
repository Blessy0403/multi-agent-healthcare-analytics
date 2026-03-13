# Pipeline Logs and Architecture

## Pipeline Architecture

### High-level flow

```
main.py
  └── run_baseline_pipeline(config, base_run_dir)     → baseline/
  └── run_multi_agent_pipeline(config, base_run_dir)    → multi_agent/
  └── PART 3: Evaluation (comparison, reports, metrics)
```

- **Entry:** `python main.py` or `python main.py diabetes` (or `heart_disease`, `framingham`).
- **Run directory:** `outputs/runs/<run_id>` where `run_id` is generated (e.g. timestamp-based) in `utils/paths.py` via `get_run_id()`.
- **Baseline pipeline** writes under `base_run_dir` (e.g. `outputs/runs/run_2025-02-24_12-00-00`).
- **Multi-agent pipeline** writes under `base_run_dir/multi_agent/`. Key artifacts are also copied to `base_run_dir/` so the dashboard can find them.

### Orchestrator (agent order)

The multi-agent pipeline is executed by `pipeline/orchestrator.py` → `execute_pipeline(config)`. Agent order is fixed:

| Step | Agent                     | Output / role |
|------|---------------------------|----------------|
| 1    | **DataAgent**             | Loads/splits data; writes train/val/test under run dir. |
| 2    | **FeatureEngineeringAgent** (if enabled) | Feature transforms; passes file_paths to Model. |
| 3    | **ModelAgent**            | Trains models; selects best; predictions; writes models, metrics, figures. |
| 4    | **ExplainabilityAgent**   | SHAP, LIME, natural language explanations; writes to `explainability/`, `figures/`. |
| 5    | **EvaluationAgent**      | Evaluation metrics (e.g. ROC-AUC) from Model + Explainability. |
| 6    | **FeedbackAgent**         | Decides accept / switch_model / retrain; can trigger Iteration 2. |
| 7    | **CrossDatasetAgent** (if enabled) | Cross-dataset validation; writes reports. |

- Artifacts are passed explicitly between agents (no shared mutable state).
- Each agent has a `run(artifacts)` method and returns a structured dict; the orchestrator merges these into a single `artifacts` dict and tracks handovers in `collaboration_metrics`.

### Runners

- **`pipeline/runner.py`**
  - **`run_baseline_pipeline(config, base_run_dir)`** – single-model baseline (e.g. `baseline/single_model_pipeline.py`); writes to `base_run_dir/baseline/` (when used).
  - **`run_multi_agent_pipeline(config, base_run_dir, baseline_results)`** – sets config paths to `base_run_dir/multi_agent/`, creates subdirs (`logs`, `models`, `explainability`, `figures`, `reports`, `data`, `research_outputs`, `dashboard_outputs`), then calls `execute_pipeline(config)` from the orchestrator. After the run, it can trigger a second iteration if the FeedbackAgent requests retrain/switch.

### Run directory layout (multi-agent)

Under `outputs/runs/<run_id>/`:

- **Root:** `metadata.json`, `logs/run.log` (and optional `run.log` under multi_agent – see below).
- **multi_agent/** (primary output of the multi-agent run):
  - `logs/`          – orchestrator and agent logs; `run.log`.
  - `models/`        – saved models (e.g. `.pkl`).
  - `data/`          – train/val/test splits (e.g. `*_train.csv`, `*_test.csv`).
  - `explainability/` – LIME, explanations JSON, etc.
  - `figures/`       – SHAP plots (e.g. `diabetes_gradient_boosting_shap_summary.png`), ROC, confusion matrix.
  - `reports/`       – feedback, reports.
  - `research_outputs/` – evaluation outputs, comparison, cross-dataset reports.
  - `dashboard_outputs/` – dashboard-specific artifacts if used.

The dashboard often resolves paths from `outputs/runs/<selected_run_id>/multi_agent/...` (e.g. `multi_agent/figures` for SHAP).

---

## Logs

### 1. Run log (main pipeline)

- **Path:** `outputs/runs/<run_id>/logs/run.log` (and for the multi-agent run, also `outputs/runs/<run_id>/multi_agent/logs/run.log` when that is the active `config.log_dir`).
- **Written by:**
  - **`main.py`** – appends short status lines via `_write_run_log(run_dir, message)` to `run_dir/logs/run.log` so the run always has some log content (even on failure):
    - At start: `[YYYY-MM-DD HH:MM:SS] Pipeline started. Run ID: <run_id>`
    - After baseline: `[timestamp] Baseline pipeline completed. Artifacts: baseline/`
    - After multi-agent: `[timestamp] Multi-agent pipeline completed. Artifacts: multi_agent/`
  - **Root logger** – `utils/logging.py` → `setup_root_logger(log_dir)` adds a `FileHandler` to `log_dir/run.log` (append mode). All `logger.info(...)` etc. from the pipeline (main, orchestrator, agents that use the root logger) go to console and to this file.
- **On failure:** `main.py` writes the exception and traceback into `config.run_dir/logs/run.log` via `_write_run_log` so the run log is never empty.

### 2. Root logger (console + run.log)

- **Setup:** `main.py` calls `setup_root_logger(config.log_dir)` at startup; the orchestrator also calls `setup_root_logger(config.log_dir)` so that during the multi-agent run, `config.log_dir` is `multi_agent/logs`, and the same root logger writes to `multi_agent/logs/run.log`.
- **Format (file):** `%(asctime)s | %(levelname)s | %(name)s | %(message)s` with datefmt `%Y-%m-%d %H:%M:%S`.
- **Format (console):** `[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s`.
- **Content:** Run ID, output dir, dataset; PART 1 (Baseline), PART 2 (Multi-agent), PART 3 (Evaluation); step headers (STEP 1: DataAgent, …); completion times; comparison/evaluation messages; errors if any.

### 3. Agent loggers (AgentLogger)

- **Class:** `utils/logging.py` → `AgentLogger(agent_name, log_dir=None)`.
- **Used by:** Orchestrator (`pipeline.orchestrator`), ExplainabilityAgent, and other agents that instantiate `AgentLogger`.
- **Behavior:**
  - Console: `[timestamp] [LEVEL] [agent_name] message`.
  - File: each agent can get a **timestamped file** in `log_dir`: `{agent_name}_{YYYYMMDD_HHMMSS}.log` (when the logger creates a file handler). The **orchestrator** and pipeline steps use the **root** logger that writes to `run.log`; agents may use both root and their own `AgentLogger`.
- **Collaboration tracking:** Each `AgentLogger` keeps a `collaboration_log` list (artifacts produced, handovers). At the end of the pipeline, the orchestrator calls `agent.logger.save_collaboration_log()` for each agent, which writes:
  - **Path:** `log_dir / {agent_name}_collaboration.json`
  - **Content:** JSON array of entries (timestamp, agent, artifact_type/path, handover events, execution time).

### 4. Dashboard and log display

- **Handover log:** The dashboard reads handover/collaboration data (e.g. from run metadata or collaboration JSONs) and shows a “Handover log” in the sidebar (`get_handover_log_entries()` in `dashboard/app.py`).
- **Run status/failure:** `dashboard/components/layout.py` uses `get_run_log_reason(run_dir, max_lines=30)` to read the last lines of `run_dir/logs/run.log` or `run_dir/multi_agent/logs/run.log` to determine run status and show a short failure reason (e.g. “Run failed or incomplete — check logs.”).

### 5. Summary table

| Log / artifact        | Location                                      | Written by / content |
|-----------------------|-----------------------------------------------|------------------------|
| Run status lines      | `outputs/runs/<run_id>/logs/run.log`          | `main.py` `_write_run_log()` – start, baseline done, multi-agent done, failure traceback |
| Pipeline run.log      | `outputs/runs/<run_id>/logs/run.log` and `.../multi_agent/logs/run.log` | Root logger – all pipeline steps, timings, errors |
| Agent collaboration  | `.../multi_agent/logs/{agent_name}_collaboration.json` | Each agent’s `AgentLogger.save_collaboration_log()` |
| Optional agent file   | `log_dir/{agent_name}_{timestamp}.log`        | AgentLogger file handler (if created) |

---

## Quick reference

- **Run directory:** `outputs/runs/<run_id>` (from `utils/paths.get_run_dir(run_id)`).
- **Pipeline flow:** `main.py` → `run_baseline_pipeline` + `run_multi_agent_pipeline` → `pipeline/runner.run_multi_agent_pipeline` → `pipeline/orchestrator.execute_pipeline` → DataAgent → (FeatureEngineeringAgent) → ModelAgent → ExplainabilityAgent → EvaluationAgent → FeedbackAgent → (CrossDatasetAgent).
- **Logs to inspect after a run:**  
  - `outputs/runs/<run_id>/logs/run.log`  
  - `outputs/runs/<run_id>/multi_agent/logs/run.log`  
  - `outputs/runs/<run_id>/multi_agent/logs/*_collaboration.json`
