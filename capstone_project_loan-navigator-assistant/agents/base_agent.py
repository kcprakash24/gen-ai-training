"""
Base agent interface and types for the loan-navigator-suite.
Defines the contract that all agents must follow.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from core.prompt_loader import prompt_loader
from core.factory import get_chat_llm


@dataclass
class AgentResponse:
    """Simplified response from any agent."""

    answer: str
    sources: list[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, prompt_source: str, temperature: float = 0.1):
        """
        Initialize the base agent with prompt loading and LLM setup.

        Args:
            prompt_source: Name of the YAML file (without extension) to load prompts from.
            temperature: Temperature setting for the chat LLM (default: 0.1).
        """
        self.prompts = prompt_loader.load_prompts(prompt_source)
        self.llm = get_chat_llm(temperature=temperature)

    @abstractmethod
    def process(self, query: Union[str, Dict[str, Any]]) -> AgentResponse:
        """
        Process a user query and return a structured response.
        """
        pass

    def handle_error(self, error_message: str) -> AgentResponse:
        """Handle errors that occur during processing."""
        error_template = self.prompts.get(
            "error_message", "An error occurred: {error_message}"
        )
        return AgentResponse(answer=error_template.format(error_message=error_message))
