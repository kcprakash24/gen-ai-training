"""Ultra-simple Key Vault helper."""

import os
from dotenv import load_dotenv
from core.config import AZURE_KEY_VAULT_URL

load_dotenv()


def get_secret(name: str) -> str:
    """
    Get secret from Key Vault or .env fallback.

    Args:
        name: Secret name (should use hyphens, e.g., AZURE-OPENAI-ENDPOINT)

    Returns:
        Secret value from Key Vault or .env

    Raises:
        ValueError: If secret not found in either Key Vault or .env
    """

    # Convert hyphens to underscores for .env lookup
    env_var_name = name.replace("-", "_").upper()

    # Try Key Vault first if URL is configured
    if AZURE_KEY_VAULT_URL:
        try:
            from azure.keyvault.secrets import SecretClient
            from azure.identity import DefaultAzureCredential

            client = SecretClient(AZURE_KEY_VAULT_URL, DefaultAzureCredential())
            secret = client.get_secret(name)
            if secret and secret.value:
                return secret.value
        except Exception as e:
            # Key Vault failed, try .env fallback
            print(f"⚠️  Key Vault unavailable for {name}, trying .env fallback: {e}")

    # Fallback to .env
    value = os.getenv(env_var_name)
    if value:
        return value

    # If we reach here, secret not found anywhere
    raise ValueError(
        f"Secret '{name}' not found in Key Vault or .env (looked for {env_var_name}). "
        f"In production, ensure Key Vault has the secret. "
        f"For local dev, add {env_var_name} to .env file."
    )
