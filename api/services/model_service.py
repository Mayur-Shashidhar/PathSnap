"""Model service for loading and managing the P2 trained ML model."""

import logging
import os
from typing import Any, Dict, List, Optional
import joblib

logger = logging.getLogger("pathsnap.model_service")


class ModelService:
    """Manages loading and serving the P2 trained machine learning model."""

    def __init__(self, model_path: str, feature_cols: List[str]):
        self.model_path = model_path
        self.feature_cols = feature_cols
        self._model: Optional[Any] = None
        self._load_error: Optional[str] = None
        self._model_info: Dict[str, Any] = {}

    @property
    def is_loaded(self) -> bool:
        """Return True if the model is successfully loaded and ready for inference."""
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        """Return the error message if model loading failed."""
        return self._load_error

    @property
    def model_info(self) -> Dict[str, Any]:
        """Return metadata about the loaded model."""
        return self._model_info

    def load(self) -> bool:
        """Load the model artifact from disk and validate its interface."""
        if not os.path.exists(self.model_path):
            self._load_error = f"Model artifact not found at: {self.model_path}"
            logger.error(self._load_error)
            self._model = None
            return False

        try:
            logger.info("Loading trained model from %s ...", self.model_path)
            model = joblib.load(self.model_path)

            # Validate prediction interface
            if not hasattr(model, "predict_proba"):
                self._load_error = (
                    f"Model of type {type(model).__name__} does not implement 'predict_proba', "
                    "which is required by the P3 HMM decoder."
                )
                logger.error(self._load_error)
                self._model = None
                return False

            # Inspect model metadata
            model_type = type(model).__name__
            n_features = getattr(model, "n_features_in_", len(self.feature_cols))

            if n_features != len(self.feature_cols):
                logger.warning(
                    "Model expects %d features, but %d feature columns are configured.",
                    n_features,
                    len(self.feature_cols)
                )

            self._model = model
            self._load_error = None
            self._model_info = {
                "model_type": model_type,
                "n_features_in": n_features,
                "feature_cols_count": len(self.feature_cols),
                "model_path": self.model_path,
            }
            logger.info("Successfully loaded %s model with %d expected features.", model_type, n_features)
            return True

        except Exception as exc:
            self._load_error = f"Failed to load model from {self.model_path}: {exc}"
            logger.exception(self._load_error)
            self._model = None
            return False

    def get_model(self) -> Any:
        """Return the loaded model instance. Raises RuntimeError if not loaded."""
        if self._model is None:
            raise RuntimeError(
                f"Model is not loaded. Error: {self._load_error or 'Model not initialized'}"
            )
        return self._model
