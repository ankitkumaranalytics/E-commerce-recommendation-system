"""
Base Recommender Interface

Defines the base class for all recommendation models.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
import joblib
from datetime import datetime
from src.utils.logger import logger


class BaseRecommender(ABC):
    """Abstract base class for recommendation models."""
    
    def __init__(self, name: str):
        """
        Initialize recommender.
        
        Args:
            name: Model name
        """
        self.name = name
        self.is_fitted = False
        self.training_date = None
        self.metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def fit(self, data: Any, **kwargs) -> None:
        """
        Train the model.
        
        Args:
            data: Training data
            **kwargs: Additional arguments
        """
        pass
    
    @abstractmethod
    def predict(self, user_id: str, n_recommendations: int = 10, **kwargs) -> List[Tuple[str, float]]:
        """
        Predict recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            **kwargs: Additional arguments
        
        Returns:
            List of (product_id, score) tuples
        """
        pass
    
    def recommend(self, user_id: str, n_recommendations: int = 10, **kwargs) -> List[str]:
        """
        Get recommendation product IDs for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            **kwargs: Additional arguments
        
        Returns:
            List of product IDs
        """
        predictions = self.predict(user_id, n_recommendations, **kwargs)
        return [product_id for product_id, _ in predictions]
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Save path
        """
        logger.info(f"Saving {self.name} to {path}")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self, path)
        
        # Save metadata
        metadata_path = path.replace('.pkl', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'name': self.name,
                'training_date': self.training_date.isoformat() if self.training_date else None,
                'is_fitted': self.is_fitted,
                'metadata': self.metadata
            }, f, indent=2)
    
    @staticmethod
    def load(path: str) -> 'BaseRecommender':
        """
        Load model from disk.
        
        Args:
            path: Model path
        
        Returns:
            Loaded model
        """
        logger.info(f"Loading model from {path}")
        return joblib.load(path)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'training_date': self.training_date,
            'metadata': self.metadata
        }
