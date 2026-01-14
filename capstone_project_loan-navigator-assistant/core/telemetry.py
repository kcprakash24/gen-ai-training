"""
Simple MLflow telemetry tracker for LLM metrics and traces.

Tracks:
- Token usage (prompt + completion)
- Fallback ratio per agent
- Citation quality (Policy Guru)
- Response times
- Detailed execution traces

KISS principle: One class, clear purpose, minimal complexity.
"""

import mlflow
import os
from typing import Optional, Dict, Any
from core.constants import MLFLOW_EXPERIMENT_NAME


class LLMTelemetry:
    """Simple LLM telemetry tracker using MLflow."""

    def __init__(self):
        """Initialize telemetry tracker."""
        # Read MLFLOW_TRACKING_URI from environment (loaded from .env or set by constants)
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        # Enable MLflow tracing
        mlflow.tracing.enable()

        self.run_id: Optional[str] = None
        self.step = 0  # Step counter for metrics

    def start_session(self, session_id: str):
        """Start tracking for a chat session."""
        if not self.run_id:
            run = mlflow.start_run(run_name=f"session_{session_id[:8]}")
            self.run_id = run.info.run_id
            self.step = 0

    def end_session(self):
        """End tracking session."""
        if self.run_id:
            mlflow.end_run()
            self.run_id = None

    def log_llm_call(
        self,
        agent_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        response_time: float,
        is_fallback: bool = False,
    ):
        """
        Log LLM call metrics.

        Args:
            agent_name: Name of agent (sql_agent, policy_guru, etc.)
            prompt_tokens: Tokens in prompt
            completion_tokens: Tokens in completion
            response_time: Response time in seconds
            is_fallback: Whether this was a fallback response
        """
        if not self.run_id:
            return

        self.step += 1

        # Log basic metrics
        mlflow.log_metrics(
            {
                f"{agent_name}.tokens.prompt": prompt_tokens,
                f"{agent_name}.tokens.completion": completion_tokens,
                f"{agent_name}.tokens.total": prompt_tokens + completion_tokens,
                f"{agent_name}.response_time": response_time,
                f"{agent_name}.fallback": 1 if is_fallback else 0,
            },
            step=self.step,
        )

    def log_citations(
        self,
        num_citations: int,
        avg_similarity: float,
    ):
        """
        Log citation quality metrics.

        Args:
            num_citations: Number of citations retrieved
            avg_similarity: Average similarity score
        """
        if not self.run_id:
            return

        mlflow.log_metrics(
            {
                "policy_guru.citations.count": num_citations,
                "policy_guru.citations.avg_similarity": avg_similarity,
            },
            step=self.step,
        )

    def log_error(self, agent_name: str, error_type: str):
        """Log error occurrence."""
        if not self.run_id:
            return

        mlflow.log_metric(f"{agent_name}.errors", 1, step=self.step)


# Global singleton instance
telemetry = LLMTelemetry()
