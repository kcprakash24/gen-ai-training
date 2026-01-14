"""
Global constants for the loan-navigator-suite project.
"""

"""
Global constants for the loan-navigator-suite project.
"""


from pathlib import Path
from core.keyvault import get_secret
from core.config import (
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    OPENAI_API_VERSION,
    ENVIRONMENT,
)

# Base directory configuration
BASE_DIR = Path(__file__).parent.parent  # Go up one level to reach project root

# File paths and directories
DATASET_DIR = BASE_DIR / "dataset"
VECTOR_STORE_DIR = BASE_DIR / "database/vector_store"
LOAN_DB_PATH = BASE_DIR / "database/loan_data/LoanDB_BlueLoans4all.sqlite"
PROMPTS_DIR = BASE_DIR / "prompts"
POLICY_DOCS = DATASET_DIR / "policy_docs"

# Vector store settings
COLLECTION_NAME = "loan_documents"
COLLECTION_METADATA = {"use_type": "PRODUCTION"}

# Document processing settings
MIN_DOCUMENT_LENGTH = 100
CHUNK_SIZE = 3500
CHUNK_OVERLAP = 500

# Azure OpenAI settings (Sensitive - from Key Vault)
AZURE_OPENAI_ENDPOINT = get_secret("AZURE-OPENAI-ENDPOINT")
AZURE_OPENAI_API_KEY = get_secret("AZURE-OPENAI-API-KEY")  # Use dash version to match Key Vault

# Model configuration
EMBEDDING_MODEL = "text-embedding-3-small"
MODEL_NAME = "gpt4o"  # Using GPT-4 for complex reasoning and calculations

# Retrieval settings
SEARCH_K = 10  # Increased to retrieve more candidate documents for better matching

# MLflow Telemetry settings
MLFLOW_TRACKING_URI = get_secret("MLFLOW-TRACKING-URI")
MLFLOW_EXPERIMENT_NAME = "team-2-capstone-loan-navigator-llm"

# Environment (from config.py)
# ENVIRONMENT - imported above
