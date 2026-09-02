"""
Cold-Start Recommendation Model

Handles recommendations for new users and new products.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from src.models.base import BaseRecommender
from src.utils.logger import logger


class ColdStartModel(BaseRecommender):
    """
    Cold-start handler for new users and products.
    
    Strategies:
    - New users: Use popular/trending products and category preferences
    - New products: Use content similarity and category popularity
    """
    
    def __init__(self, name: str = "cold_start"):
        """Initialize cold-start model."""
        super().__init__(name)
        self.popular_products: List[str] = None
        self.trending_products: List[str] = None
        self.category_popularity: Dict[str, List[str]] = None
        self.products_df: Optional[pd.DataFrame] = None
    
    def fit(self, interactions: pd.DataFrame, products: pd.DataFrame,
            n_popular: int = 20, n_trending: int = 10) -> None:
        """
        Prepare cold-start data.
        
        Args:
            interactions: Interactions dataframe
            products: Products dataframe
            n_popular: Number of popular products to track
            n_trending: Number of trending products to track
        """
        logger.info(f"Setting up {self.name} model")
        
        self.products_df = products.copy()
        
        # Popular products
        interaction_counts = interactions.groupby('product_id').size()
        self.popular_products = (
            interaction_counts.nlargest(n_popular).index.tolist()
        )
        
        # Trending products (last 7 days)
        interactions['timestamp'] = pd.to_datetime(interactions['timestamp'])
        from datetime import timedelta
        cutoff = pd.Timestamp.now() - timedelta(days=7)
        recent = interactions[interactions['timestamp'] >= cutoff]
        recent_counts = recent.groupby('product_id').size()
        self.trending_products = (
            recent_counts.nlargest(n_trending).index.tolist()
        )
        
        # Category popularity
        self.category_popularity = {}
        for category in products['category'].unique():
            category_products = products[products['category'] == category]['product_id'].tolist()
            category_interactions = interaction_counts[
                interaction_counts.index.isin(category_products)
            ].nlargest(10).index.tolist()
            self.category_popularity[category] = category_interactions
        
        self.is_fitted = True
        self.training_date = datetime.now()
        self.metadata = {
            'n_popular': len(self.popular_products),
            'n_trending': len(self.trending_products),
            'n_categories': len(self.category_popularity)
        }
        
        logger.info("Cold-start setup complete")
    
    def predict(self, user_id: str, n_recommendations: int = 10,
                user_profile: Optional[Dict[str, Any]] = None,
                **kwargs) -> List[Tuple[str, float]]:
        """
        Get cold-start recommendations for new users.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            user_profile: Optional user profile with preferences
            **kwargs: Additional arguments
        
        Returns:
            List of (product_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")
        
        recommendations = []
        
        # 1. Add trending products (high weight)
        for idx, pid in enumerate(self.trending_products[:n_recommendations // 2]):
            score = 1.0 - (idx / len(self.trending_products))
            recommendations.append((pid, score * 0.7))
        
        # 2. Add category-based if user has preferences
        if user_profile and 'preferred_categories' in user_profile:
            for category in user_profile['preferred_categories']:
                if category in self.category_popularity:
                    products = self.category_popularity[category]
                    for idx, pid in enumerate(products[:2]):
                        score = 1.0 - (idx / len(products))
                        recommendations.append((pid, score * 0.5))
        
        # 3. Add popular products to fill
        for idx, pid in enumerate(self.popular_products):
            if len(recommendations) < n_recommendations:
                score = 1.0 - (idx / len(self.popular_products))
                recommendations.append((pid, score * 0.3))
        
        # Remove duplicates, keeping highest score
        seen = {}
        for pid, score in recommendations:
            if pid not in seen or score > seen[pid]:
                seen[pid] = score
        
        # Sort and return
        result = sorted(seen.items(), key=lambda x: x[1], reverse=True)
        return result[:n_recommendations]
    
    def get_new_product_recommendations(self, product_id: str,
                                       n_recommendations: int = 10,
                                       content_model: Optional[BaseRecommender] = None,
                                       **kwargs) -> List[Tuple[str, float]]:
        """
        Get recommendations for a new product.
        
        Args:
            product_id: New product ID
            n_recommendations: Number of recommendations
            content_model: Optional content-based model for similarity
            **kwargs: Additional arguments
        
        Returns:
            List of (related_product_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")
        
        if product_id not in self.products_df['product_id'].values:
            logger.warning(f"Product {product_id} not found")
            return []
        
        product = self.products_df[self.products_df['product_id'] == product_id].iloc[0]
        category = product['category']
        
        # Get products from same category
        category_products = self.products_df[
            (self.products_df['category'] == category) & 
            (self.products_df['product_id'] != product_id)
        ]['product_id'].tolist()
        
        # Use content model if available, otherwise use category popularity
        if content_model and hasattr(content_model, 'predict'):
            try:
                recommendations = content_model.predict(
                    user_id='cold_start_product',
                    n_recommendations=n_recommendations,
                    user_interactions=[product_id],
                    **kwargs
                )
                return recommendations
            except Exception as e:
                logger.warning(f"Content model failed: {e}, using category fallback")
        
        # Fallback: use category popularity
        if category in self.category_popularity:
            popular = self.category_popularity[category]
            recommendations = [
                (pid, 1.0 - (idx / len(popular)))
                for idx, pid in enumerate(popular[:n_recommendations])
                if pid != product_id
            ]
            return recommendations
        
        return []
