"""Shared helper for building a full AgentState dict in tests — avoids re-typing every
key (and silently drifting out of sync with state.py) across eval/integration tests."""


def make_state(dataset_path: str, target: str, problem_type: str, llm_config: dict, **overrides) -> dict:
    state = {
        "llm_config": llm_config,
        "job_id": "eval-job",
        "dataset_path": dataset_path,
        "user_description": f"Predict {target}",
        "target_column": target,
        "problem_type": problem_type,
        "profile": None, "df_sample_path": None, "original_columns": [],
        "eda_findings": [], "eda_charts": [], "target_analysis": None, "eda_narrative": None,
        "features_applied": [], "processed_dataset_path": None, "preprocessor_path": None,
        "feature_names": [], "dropped_columns": [], "plan": None,
        "experiments": [], "baseline_score": None, "critic_history": [], "retry_count": 0,
        "best_model_name": None, "best_model_path": None, "evaluation_metrics": None,
        "evaluation_charts": [], "model_explanation": None, "statistical_comparison": None,
        "shap_summary": None, "report_pdf_path": None, "report_json": None,
        "current_agent": "starting", "status": "running", "progress_pct": 0,
        "messages": [], "errors": [], "warnings": [], "token_usage": [], "trace_events": [],
    }
    state.update(overrides)
    return state
