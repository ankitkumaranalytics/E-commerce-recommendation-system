"""
Hybrid Recommendation Model

Combines multiple recommendation models for better results.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from src.models.base import BaseRecommender
from src.utils.logger import logger
from src.utils.common import normalize_scores, combine_scores


class HybridModel(BaseRecommender):
    """
    Hybrid recommender combining multiple models.
    
    Combines scores from:
    - Collaborative filtering
    - Content-based
    - Popularity
    - Trending
    """
    
    def __init__(self, name: str = "hybrid"):
        """Initialize hybrid model."""
        super().__init__(name)
        self.models: Dict[str, BaseRecommender] = {}
        self.weights: Dict[str, float] = {}
    
    def add_model(self, model_name: str, model: BaseRecommender, weight: float) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            model_name: Name for the model in ensemble
            model: Recommender model instance
            weight: Weight for this model's scores
        """
        self.models[model_name] = model
        self.weights[model_name] = weight
        logger.debug(f"Added model {model_name} with weight {weight}")
    
    def fit(self, **kwargs) -> None:
        """
        Fit is not needed for hybrid - sub-models should be trained separately.
        """
        logger.info(f"Setting up {self.name} model")
        
        # Verify all models are fitted
        unfitted = [name for name, model in self.models.items() if not model.is_fitted]
        if unfitted:
            logger.warning(f"Some models not fitted: {unfitted}")
        
        self.is_fitted = True
        self.training_date = datetime.now()
        self.metadata = {
            'n_models': len(self.models),
            'model_names': list(self.models.keys()),
            'weights': self.weights
        }
    
    def predict(self, user_id: str, n_recommendations: int = 10,
                user_interactions: List[str] = None,
                exclude_products: List[str] = None,
                products: Optional[pd.DataFrame] = None,
                **kwargs) -> List[Tuple[str, float]]:
        """
        Get hybrid recommendations combining all models.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            user_interactions: List of products user has interacted with
            exclude_products: Products to exclude
            products: Products dataframe for business rules
            **kwargs: Additional arguments
        
        Returns:
            List of (product_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")
        
        if not self.models:
            logger.warning("No models in hybrid ensemble")
            return []
        
        # Collect scores from all models
        all_scores: Dict[str, Dict[str, float]] = {}
        all_products: set = set()
        
        for model_name, model in self.models.items():
            try:
                if model.is_fitted:
                    scores = model.predict(
                        user_id,
                        n_recommendations=n_recommendations * 2,  # Get more candidates
                        user_interactions=user_interactions,
                        **kwargs
                    )
                    
                    all_scores[model_name] = {pid: score for pid, score in scores}
                    all_products.update([pid for pid, _ in scores])
            except Exception as e:
                logger.warning(f"Error in {model_name}: {str(e)}")
        
        if not all_products:
            logger.debug(f"No predictions from any model for user {user_id}")
            return []
        
        # Combine scores
        combined_scores = {}
        for product_id in all_products:
            product_scores = {}
            for model_name, scores in all_scores.items():
                product_scores[model_name] = scores.get(product_id, 0.0)
            
            # Apply weights and sum
            combined = sum(
                product_scores.get(name, 0) * self.weights.get(name, 0)
                for name in self.models.keys()
            )
            
            # Normalize by sum of available weights
            available_weight = sum(
                self.weights.get(name, 0) for name in self.models.keys()
                if product_scores.get(name, 0) > 0
            )
            
            if available_weight > 0:
                combined_scores[product_id] = combined / available_weight
            else:
                combined_scores[product_id] = combined
        
        # Apply business rules
        if products is not None and exclude_products is None:
            exclude_products = []
        
        if exclude_products:
            combined_scores = {
                pid: score for pid, score in combined_scores.items()
                if pid not in exclude_products
            }
        
        # Apply product filters if dataframe provided
        if products is not None:
            # Filter out-of-stock
            in_stock_products = set(products[products['stock'] > 0]['product_id'].values)
            combined_scores = {
                pid: score for pid, score in combined_scores.items()
                if pid in in_stock_products
            }
            
            # Filter by minimum rating
            min_rating_products = set(products[products['rating'] >= 2.0]['product_id'].values)
            combined_scores = {
                pid: score for pid, score in combined_scores.items()
                if pid in min_rating_products
            }
        
        # Sort by combined score
        ranked = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return ranked[:n_recommendations]
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get detailed ensemble information."""
        return {
            'name': self.name,
            'n_models': len(self.models),
            'models': {
                name: {
                    'type': model.__class__.__name__,
                    'weight': self.weights.get(name, 0),
                    'is_fitted': model.is_fitted,
                    'training_date': model.training_date
                }
                for name, model in self.models.items()
            },
            'weights': self.weights,
            'total_weight': sum(self.weights.values())
        }
