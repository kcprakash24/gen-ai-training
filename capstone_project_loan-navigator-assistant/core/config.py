"""
Application configuration - Non-sensitive values safe to commit.
"""

import os

# Azure Key Vault Configuration
# Can be overridden by environment variable (set to empty string to disable for local Docker)
AZURE_KEY_VAULT_URL = os.getenv(
    "AZURE_KEY_VAULT_URL", "https://team-2-capstone.vault.azure.net/"
)

# Azure OpenAI Configuration (Non-sensitive)
AZURE_OPENAI_API_VERSION = "2024-08-01-preview"
AZURE_OPENAI_DEPLOYMENT = "gpt4o"
OPENAI_API_VERSION = "2024-08-01-preview"

# Environment
ENVIRONMENT = "production"  # Change to "development" for local testing
