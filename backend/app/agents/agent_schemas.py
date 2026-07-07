from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class ColumnDropDecision(BaseModel):
    column: str = Field(description="Exact column name to drop")
    reason: str = Field(description="Why this column should be dropped (e.g. leakage, ID, redundant)")


class AnalysisPlan(BaseModel):
    """The Planner agent's strategy for this dataset — decided by the LLM, not hardcoded."""

    model_config = ConfigDict(protected_namespaces=())

    reasoning: str = Field(description="2-4 sentences explaining the overall analysis strategy")
    columns_to_drop: list[ColumnDropDecision] = Field(
        default_factory=list,
        description="Columns to drop beyond what deterministic rules already catch — "
                    "typically leakage columns flagged in EDA findings",
    )
    model_shortlist: list[str] = Field(
        description="Subset and priority order of model keys to train, chosen for this dataset's "
                    "size/shape (e.g. skip heavy boosters on tiny datasets to avoid overfitting)"
    )
    risk_flags: list[str] = Field(
        default_factory=list,
        description="Concrete risks for this run (e.g. 'severe class imbalance', 'small sample size')",
    )


class CriticVerdict(BaseModel):
    """The Critic agent's review of training results — can send the run back for another pass."""

    model_config = ConfigDict(protected_namespaces=())

    verdict: Literal["approve", "retry_features", "retry_models", "insufficient_signal"] = Field(
        description="approve: proceed to evaluation. retry_features: go back and re-plan feature "
                    "engineering. retry_models: go back and try a different model shortlist. "
                    "insufficient_signal: the data has no real predictive signal — approve but flag it."
    )
    confidence: Literal["high", "medium", "low"] = Field(description="Confidence in this verdict")
    reasoning: str = Field(description="2-4 sentences justifying the verdict")
    concerns: list[str] = Field(default_factory=list, description="Specific concerns, if any")
    additional_columns_to_drop: list[ColumnDropDecision] = Field(
        default_factory=list,
        description="Only used when verdict=retry_features: extra columns the Planner missed "
                    "that should also be dropped (e.g. a leakage column that slipped through)",
    )
    revised_model_shortlist: list[str] = Field(
        default_factory=list,
        description="Only used when verdict=retry_models: a different set/order of model keys "
                    "to try instead of the ones already trained",
    )
