# --- Ensure Key Vault secrets are mapped to env vars before anything else ---
import project_bootstrap
"""
Main FastAPI application entry point.
"""

import os
import sys
from pathlib import Path

# === CENTRALIZED PYCACHE SETUP (BEFORE ANY OTHER IMPORTS) ===
project_root = Path(__file__).parent
cache_dir = project_root / ".pycache"
cache_dir.mkdir(exist_ok=True)

# Set environment variable for this and future Python processes
os.environ["PYTHONPYCACHEPREFIX"] = str(cache_dir)

# For Python 3.8+, try to set sys.pycache_prefix directly
if hasattr(sys, "pycache_prefix"):
    if not sys.pycache_prefix:
        # This works for modules imported AFTER this point
        sys.pycache_prefix = str(cache_dir)
        print(f"✅ Set sys.pycache_prefix to: {sys.pycache_prefix}")
    else:
        print(f"✅ sys.pycache_prefix already set to: {sys.pycache_prefix}")
else:
    print(f"✅ PYTHONPYCACHEPREFIX set to: {cache_dir}")

from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=project_root / ".env")


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.chat import router as chat_router

# Create FastAPI app
app = FastAPI(
    title="Loan Navigator API",
    description="API for loan policy analysis and navigation",
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],  # Streamlit default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chat_router)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
