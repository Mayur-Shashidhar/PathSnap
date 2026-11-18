"""PathSnap API Configuration."""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base repository root directory
REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Metadata
    API_TITLE: str = "PathSnap API"
    API_DESCRIPTION: str = "Distributed Map-Matching & Route Reconstruction Backend (Laptop 1)"
    API_VERSION: str = "1.0.0"

    # Server Host and Port
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Model Configuration
    # Defaults to the final trained LightGBM baseline model
    MODEL_PATH: str = str(REPO_ROOT / "models" / "lightgbm_baseline.joblib")
    
    # Feature columns expected by the trained model (in exact training order)
    FEATURE_COLS: List[str] = [
        "dist", "rank", "dist_minus_min", "dist_ratio",
        "n_cands", "proj_t", "heading_diff",
        "heading_diff_undirected", "heading_valid",
        "speed", "dt_prev", "dt_next", "arc_length",
        "out_degree", "in_degree", "dist_prev_pt",
        "dist_next_pt", "in_prev_cands", "in_next_cands"
    ]

    # Evaluator Data Directory (reserved for organizer evaluation data)
    # Never defaults to training data. Configurable via PATHSNAP_EVALUATOR_DIR.
    EVALUATOR_DATA_DIR: str = str(REPO_ROOT / "data" / "raw" / "evaluator")

    # HMM / Viterbi parameters
    W_EMIT: float = 1.0

    # Evaluation Thresholds
    RELIABLE_DIST_THRESHOLD: float = 100.0  # metres (matches P1 threshold)

    # CORS configuration for UI cross-laptop connectivity
    CORS_ORIGINS: List[str] = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="PATHSNAP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global cached settings instance
settings = Settings()
