"""
Simple decorator for automatic telemetry tracking with MLflow tracing.

DRY principle: Write once, use everywhere.
"""

import time
import mlflow
from functools import wraps
from typing import Callable, Any
from core.telemetry import telemetry


def track_agent(agent_name: str) -> Callable:
    """
    Decorator to automatically track agent calls with MLflow tracing.

    Usage:
        @track_agent("sql_agent")
        def process(self, query):
            return response

    Automatically tracks:
    - Response time
    - Token usage (if available)
    - Fallback status
    - Detailed execution trace
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        @mlflow.trace(name=agent_name, span_type="AGENT")
        def wrapper(self, *args, **kwargs) -> Any:
            start_time = time.time()

            try:
                # Execute function
                result = func(self, *args, **kwargs)

                # Calculate metrics
                response_time = time.time() - start_time
                prompt_tokens = 0
                completion_tokens = 0
                is_fallback = False

                # Extract metrics from response
                if hasattr(result, "metadata") and result.metadata:
                    is_fallback = result.metadata.get(
                        "is_fallback", False
                    ) or result.metadata.get("sql_fallback_flag", False)

                    # Get token usage if available
                    usage = result.metadata.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)

                # Estimate tokens if not tracked (1 token ≈ 4 chars)
                if (
                    completion_tokens == 0
                    and hasattr(result, "answer")
                    and result.answer
                ):
                    completion_tokens = len(result.answer) // 4

                # Log to MLflow
                telemetry.log_llm_call(
                    agent_name=agent_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    response_time=response_time,
                    is_fallback=is_fallback,
                )

                return result

            except Exception as e:
                # Log error
                telemetry.log_error(agent_name, type(e).__name__)
                raise

        return wrapper

    return decorator
