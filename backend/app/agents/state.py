from typing import Literal, Optional, TypedDict


class DatasetProfile(TypedDict):
    n_rows: int
    n_cols: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    missing_pct: float
    duplicate_rows: int
    memory_mb: float
    col_profiles: dict
    is_large: bool
    sample_size: int


class AgentState(TypedDict):
    # Auth — passed per-request, never persisted
    llm_config: dict  # {api_key, provider, model} — NEVER log this

    # Input
    job_id: str
    dataset_path: str
    user_description: str
    target_column: Optional[str]
    problem_type: Optional[Literal["classification", "regression", "clustering", "unknown"]]

    # Dataset metadata
    profile: Optional[DatasetProfile]
    df_sample_path: Optional[str]
    original_columns: list

    # EDA outputs
    eda_findings: list  # [{finding, severity: info|warning|critical, column}]
    eda_charts: list    # [{title, chart_type, base64_png, description}]
    target_analysis: Optional[dict]
    eda_narrative: Optional[dict]  # LLM-generated summary

    # Feature engineering outputs
    features_applied: list  # [{name, type, rationale, columns_affected}]
    processed_dataset_path: Optional[str]
    preprocessor_path: Optional[str]  # path to unfitted sklearn ColumnTransformer (joblib)
    feature_names: list
    dropped_columns: list

    # Planner outputs (agentic strategy — decided by LLM, not hardcoded)
    plan: Optional[dict]  # AnalysisPlan.model_dump(): reasoning, columns_to_drop, model_shortlist, ...

    # Experiment outputs
    experiments: list  # [{model_name, params, cv_scores, train_time_s, mlflow_run_id, is_baseline}]
    baseline_score: Optional[dict]  # dummy model's primary_score, for statistical comparison

    # Critic outputs (reflection loop — can send the run back to features/train)
    critic_history: list  # [{verdict, confidence, reasoning, concerns, retry_count}]
    retry_count: int

    # Evaluation outputs
    best_model_name: Optional[str]
    best_model_path: Optional[str]
    evaluation_metrics: Optional[dict]
    evaluation_charts: list
    model_explanation: Optional[str]  # LLM-generated non-technical explanation
    statistical_comparison: Optional[dict]  # bootstrap CI + lift vs baseline
    shap_summary: Optional[dict]  # SHAP-based explainability

    # Report
    report_pdf_path: Optional[str]
    report_json: Optional[dict]

    # Control
    current_agent: str
    status: Literal["running", "completed", "failed"]
    progress_pct: int
    messages: list   # streamed live to frontend via SSE (human-readable log lines)
    errors: list
    warnings: list

    # Observability
    token_usage: list  # [{agent, model, input_tokens, output_tokens, cost_usd, latency_s}]
    trace_events: list  # [{kind, agent, payload, ts}] — thought/tool_call/plan/critic/retry/token_usage
