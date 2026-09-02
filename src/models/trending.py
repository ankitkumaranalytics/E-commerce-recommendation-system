"""
Trending Products Recommendation Model

Recommends products that are gaining popularity recently.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
from src.models.base import BaseRecommender
from src.utils.logger import logger


class TrendingModel(BaseRecommender):
    """
    Trending recommender.
    
    Recommends products with increasing recent interactions.
    """
    
    def __init__(self, name: str = "trending"):
        """Initialize trending model."""
        super().__init__(name)
        self.product_scores: Dict[str, float] = None
    
    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame,
            window_days: int = 7, min_interactions: int = 5) -> None:
        """
        Train trending model.
        
        Args:
            interactions: Interactions dataframe
            products: Products dataframe
            window_days: Window for recent interactions
            min_interactions: Minimum interactions to be trending
        """
        logger.info(f"Training {self.name} model (window={window_days} days)")
        
        interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
        
        # Filter recent interactions
        cutoff_date = pd.Timestamp.now() - timedelta(days=window_days)
        recent = interactions[interactions['timestamp'] >= cutoff_date]
        
        # Count recent interactions
        interaction_counts = recent.groupby('product_id').size()
        
        # Filter by minimum interactions
        trending_products = interaction_counts[interaction_counts >= min_interactions]
        
        # Calculate trending score (higher is more trending)
        self.product_scores = {}
        
        for product_id in products['product_id']:
            if product_id in trending_products.index:
                # Normalize recent interactions
                recent_count = trending_products[product_id]
                max_count = trending_products.max()
                score = recent_count / (max_count + 1e-10)
                self.product_scores[product_id] = score
            else:
                self.product_scores[product_id] = 0.0
        
        self.is_fitted = True
        self.training_date = datetime.now()
        self.metadata = {
            'window_days': window_days,
            'min_interactions': min_interactions,
            'n_trending_products': len(trending_products)
        }
        
        logger.info(f"Found {len(trending_products)} trending products")
    
    def predict(self, user_id: str, n_recommendations: int = 10,
                **kwargs) -> List[Tuple[str, float]]:
        """
        Get trending product recommendations.
        
        Args:
            user_id: User ID (not used for trending)
            n_recommendations: Number of recommendations
            **kwargs: Additional arguments
        
        Returns:
            List of (product_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Sort by score and return top-k
        sorted_products = sorted(
            self.product_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter out products with zero score
        sorted_products = [
            (pid, score) for pid, score in sorted_products
            if score > 0
        ]
        
        return sorted_products[:n_recommendations]
