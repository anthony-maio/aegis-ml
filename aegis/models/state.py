from pydantic import BaseModel, Field
from typing import Literal, Optional, Any
from datetime import datetime, timezone


class TrainingSpec(BaseModel):
    """User's training intent - extracted by LLM, validated by profiler."""
    method: Literal["full_finetune", "lora", "qlora"]
    model_name: str
    dataset_path: str
    num_epochs: int = 3
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    seq_len: int = 512
    target_gpu: str = "a10g"
    streaming: bool = False
    max_steps: int = 0  # 0 = use num_epochs, >0 = override with fixed step count
    trust_remote_code: bool = False


class BudgetPolicy(BaseModel):
    """FinOps gate configuration."""
    max_budget_usd: float = 5.00
    soft_threshold_usd: float = 2.00
    allow_auto_approve: bool = True


class CostEstimate(BaseModel):
    """Output of cost profiler."""
    estimated_cost_usd: float
    estimated_vram_gb: float
    estimated_duration_min: float
    cost_breakdown: dict[str, Any]


class RunEvent(BaseModel):
    """Append-only log for auditability."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    phase: str
    status: Literal["started", "completed", "failed", "interrupted"]
    message: str
    data: Optional[dict[str, Any]] = None


class AegisState(BaseModel):
    """Complete graph state - immutable updates via .model_copy()."""
    user_input: Optional[str] = None
    spec: Optional[TrainingSpec] = None
    budget_policy: BudgetPolicy = Field(default_factory=BudgetPolicy)
    cost_estimate: Optional[CostEstimate] = None
    events: list[RunEvent] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    generated_code: Optional[str] = None
    execution_result: Optional[dict[str, Any]] = None
    eval_result: Optional[dict[str, Any]] = None
    human_decision: Optional[Literal["approve", "optimize", "cancel"]] = None
    final_report: Optional[str] = None
