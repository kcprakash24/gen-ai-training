"""
Factory pattern for creating and managing application instances.
Provides centralized object creation and singleton behavior for expensive resources.
"""

from typing import Optional, Dict, Any
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from .constants import MODEL_NAME, EMBEDDING_MODEL


class LLMFactory:
    """Factory for creating and managing LLM instances with singleton behavior."""

    _chat_instances: Dict[str, AzureChatOpenAI] = {}
    _embedding_instances: Dict[str, AzureOpenAIEmbeddings] = {}

    @classmethod
    def get_chat_llm(
        cls, model: str = MODEL_NAME, temperature: float = 0.7, **kwargs
    ) -> AzureChatOpenAI:
        """Get or create a chat LLM instance with singleton behavior."""
        # Create a unique key based on model and key parameters
        key = f"{model}_{temperature}_{hash(frozenset(kwargs.items()))}"

        if key not in cls._chat_instances:
            cls._chat_instances[key] = AzureChatOpenAI(
                model=model, temperature=temperature, **kwargs
            )

        return cls._chat_instances[key]

    @classmethod
    def get_embedding_llm(
        cls, model: str = EMBEDDING_MODEL, **kwargs
    ) -> AzureOpenAIEmbeddings:
        """Get or create an embedding LLM instance with singleton behavior."""
        key = f"{model}_{hash(frozenset(kwargs.items()))}"

        if key not in cls._embedding_instances:
            cls._embedding_instances[key] = AzureOpenAIEmbeddings(model=model, **kwargs)

        return cls._embedding_instances[key]


class AgentFactory:
    """Factory for creating and managing agent instances."""

    _agent_instances: Dict[str, Any] = {}

    @classmethod
    def get_policy_guru(cls):
        """Get or create PolicyGuru agent instance."""
        if "policy_guru" not in cls._agent_instances:
            # Import here to avoid circular imports
            from agents.policy_guru import PolicyGuru

            cls._agent_instances["policy_guru"] = PolicyGuru()

        return cls._agent_instances["policy_guru"]

    @classmethod
    def get_what_if_calculator(cls):
        """Get or create WhatIfCalculator agent instance."""
        if "what_if_calculator" not in cls._agent_instances:
            # Import here to avoid circular imports
            from agents.what_if_calculator import WhatIfCalculator

            cls._agent_instances["what_if_calculator"] = WhatIfCalculator()

        return cls._agent_instances["what_if_calculator"]

    @classmethod
    def get_sql_agent(cls):
        """Get or create SQLAgent instance."""
        if "sql_agent" not in cls._agent_instances:
            # Import here to avoid circular imports
            from agents.sql_agent import SQLAgent

            cls._agent_instances["sql_agent"] = SQLAgent()

        return cls._agent_instances["sql_agent"]


# Convenience functions for easy access
def get_chat_llm(temperature: float = 0.7, **kwargs) -> AzureChatOpenAI:
    """Convenience function to get chat LLM."""
    return LLMFactory.get_chat_llm(temperature=temperature, **kwargs)


def get_embedding_llm(**kwargs) -> AzureOpenAIEmbeddings:
    """Convenience function to get embedding LLM."""
    return LLMFactory.get_embedding_llm(**kwargs)


def get_policy_guru():
    """Convenience function to get PolicyGuru agent."""
    return AgentFactory.get_policy_guru()


def get_what_if_calculator():
    """Convenience function to get WhatIfCalculator agent."""
    return AgentFactory.get_what_if_calculator()


def get_sql_agent():
    """Convenience function to get SQLAgent."""
    return AgentFactory.get_sql_agent()


def get_workflow_engine():
    """Convenience function to get WorkflowEngine."""
    from core.workflow_engine import WorkflowEngine

    return WorkflowEngine()
