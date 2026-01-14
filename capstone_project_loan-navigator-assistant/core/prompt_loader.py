"""
Prompt loader utility for loading prompts from YAML files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List


class PromptLoader:
    """Loads and manages prompts from YAML files."""

    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}

    def load_prompts(self, agent_name: str) -> Dict[str, Any]:
        """Load all prompts for a specific agent from YAML file."""
        if agent_name in self._cache:
            return self._cache[agent_name]

        yaml_file = self.prompts_dir / f"{agent_name}.yaml"

        if not yaml_file.exists():
            raise FileNotFoundError(f"Prompt file not found: {yaml_file}")

        with open(yaml_file, "r", encoding="utf-8") as file:
            prompts = yaml.safe_load(file)

        self._cache[agent_name] = prompts
        return prompts


# Global instance for easy access
prompt_loader = PromptLoader()
