
# --- Bootstrap Key Vault secrets to environment variables as early as possible ---
import os
from core.keyvault import get_secret

# List all secrets that need to be mapped from Key Vault to env vars
SECRETS = [
    "AZURE-OPENAI-API-KEY",
    "AZURE-OPENAI-ENDPOINT",
    "MLFLOW-TRACKING-URI",
    "OPENAI-API-VERSION",
]
for dash_name in SECRETS:
    env_name = dash_name.replace("-", "_").upper()
    try:
        value = get_secret(dash_name)
        if value:
            os.environ[env_name] = value
    except Exception:
        pass

import sys
import subprocess
import shutil
from pathlib import Path


def main():
    # Import after path setup
    from core.constants import BASE_DIR, DATASET_DIR, LOAN_DB_PATH
    
    # Create loan_data directory
    LOAN_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy database file
    source_db = DATASET_DIR / "LoanDB_BlueLoans4all.sqlite"
    if source_db.exists() and not LOAN_DB_PATH.exists():
        shutil.copy2(source_db, LOAN_DB_PATH)
    
    # Setup pycache
    cache_dir = BASE_DIR / ".pycache"
    cache_dir.mkdir(exist_ok=True)
    os.environ["PYTHONPYCACHEPREFIX"] = str(cache_dir)

    # Run document embedder
    subprocess.run([sys.executable, "-m", "scripts.document_embedder"], env=os.environ)


if __name__ == "__main__":
    main()
