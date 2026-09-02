"""
Popularity-Based Recommendation Model

Recommends popular products based on interactions and ratings.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime
from src.models.base import BaseRecommender
from src.utils.logger import logger


class PopularityModel(BaseRecommender):
    """
    Popularity-based recommender.
    
    Recommends products based on:
    - Total interactions
    - Average rating
    - Recent popularity (time decay)
    """
    
    def __init__(self, name: str = "popularity"):
        """Initialize popularity model."""
        super().__init__(name)
        self.product_scores: pd.Series = None
    
    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame, 
            time_decay_enabled: bool = True, half_life_days: int = 30) -> None:
        """
        Train popularity model.
        
        Args:
            interactions: Interactions dataframe
            products: Products dataframe
            time_decay_enabled: Apply time decay to recent interactions
            half_life_days: Half-life for time decay
        """
        logger.info(f"Training {self.name} model")
        
        # Count interactions per product
        interaction_counts = interactions.groupby('product_id').size()
        
        # Apply time decay if enabled
        if time_decay_enabled:
            interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
            current_time = pd.Timestamp.now()
            interactions['days_ago'] = (current_time - interactions['timestamp']).dt.days
            interactions['decay_weight'] = 2 ** (-interactions['days_ago'] / half_life_days)
            
            interaction_counts = interactions.groupby('product_id')['decay_weight'].sum()
        
        # Normalize interaction counts
        interaction_scores = (
            (interaction_counts - interaction_counts.min()) / 
            (interaction_counts.max() - interaction_counts.min() + 1e-10)
        )
        
        # Get average rating
        avg_ratings = products.groupby('product_id')['rating'].mean()
        rating_scores = avg_ratings / 5.0  # Normalize to [0, 1]
        
        # Combine scores: 70% interactions, 30% rating
        self.product_scores = {}
        for product_id in products['product_id']:
            interaction_score = interaction_scores.get(product_id, 0)
            rating_score = rating_scores.get(product_id, 0)
            combined = (0.7 * interaction_score + 0.3 * rating_score)
            self.product_scores[product_id] = combined
        
        self.is_fitted = True
        self.training_date = datetime.now()
        self.metadata = {
            'time_decay_enabled': time_decay_enabled,
            'half_life_days': half_life_days,
            'n_products': len(self.product_scores)
        }
        
        logger.info(f"Training complete. Scored {len(self.product_scores)} products")
    
    def predict(self, user_id: str, n_recommendations: int = 10, 
                **kwargs) -> List[Tuple[str, float]]:
        """
        Get popular product recommendations.
        
        Args:
            user_id: User ID (not used for popularity)
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
        
        return sorted_products[:n_recommendations]
