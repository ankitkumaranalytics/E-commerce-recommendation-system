"""
Recommendation Engine

Central orchestration of all recommendation models and business rules.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
import json
from src.utils.logger import logger
from src.utils.common import remove_duplicates, SimpleCache
from src.models import (
    PopularityModel,
    ContentBasedModel,
    CollaborativeFilteringModel,
    TrendingModel,
    HybridModel,
    BaseRecommender
)
from src.models.cold_start import ColdStartModel


class RecommendationEngine:
    """
    Central recommendation engine.
    
    Orchestrates multiple models and applies business rules.
    """
    
    def __init__(self):
        """Initialize recommendation engine."""
        self.popularity_model: Optional[PopularityModel] = None
        self.content_model: Optional[ContentBasedModel] = None
        self.collab_model: Optional[CollaborativeFilteringModel] = None
        self.trending_model: Optional[TrendingModel] = None
        self.cold_start_model: Optional[ColdStartModel] = None
        self.hybrid_model: Optional[HybridModel] = None
        
        self.users_df: Optional[pd.DataFrame] = None
        self.products_df: Optional[pd.DataFrame] = None
        self.interactions_df: Optional[pd.DataFrame] = None
        
        # Cache
        self.cache = SimpleCache(max_size=1000, ttl_seconds=3600)
        self.is_ready = False
    
    def initialize_models(self,
                         users: pd.DataFrame,
                         products: pd.DataFrame,
                         interactions: pd.DataFrame,
                         weights: Dict[str, float] = None) -> None:
        """
        Initialize and train all models.
        
        Args:
            users: Users dataframe
            products: Products dataframe
            interactions: Interactions dataframe
            weights: Weights for hybrid model
        """
        logger.info("Initializing recommendation engine")
        
        self.users_df = users.copy()
        self.products_df = products.copy()
        self.interactions_df = interactions.copy()
        
        # Parse dates
        self.interactions_df['timestamp'] = pd.to_datetime(
            self.interactions_df['timestamp']
        )
        self.products_df['created_at'] = pd.to_datetime(
            self.products_df['created_at']
        )
        
        try:
            # Popularity model
            logger.info("Training popularity model...")
            self.popularity_model = PopularityModel()
            self.popularity_model.fit(interactions, products)
            
            # Content-based model
            logger.info("Training content-based model...")
            self.content_model = ContentBasedModel()
            self.content_model.fit(products, interactions)
            
            # Collaborative filtering model
            logger.info("Training collaborative filtering model...")
            from src.features.engineering import FeatureEngineer
            matrix, users_list, products_list = FeatureEngineer.create_interaction_matrix(
                interactions, users, products
            )
            
            self.collab_model = CollaborativeFilteringModel(algorithm='svd', n_factors=50)
            self.collab_model.fit(matrix, users_list, products_list)
            
            # Trending model
            logger.info("Training trending model...")
            self.trending_model = TrendingModel()
            self.trending_model.fit(interactions, products, window_days=7, min_interactions=3)
            
            # Cold-start model
            logger.info("Setting up cold-start model...")
            self.cold_start_model = ColdStartModel()
            self.cold_start_model.fit(interactions, products)
            
            # Hybrid model
            logger.info("Setting up hybrid model...")
            self.hybrid_model = HybridModel()
            
            if weights is None:
                weights = {
                    'collaborative': 0.40,
                    'content_based': 0.25,
                    'popularity': 0.15,
                    'trending': 0.10,
                    'business_rules': 0.10
                }
            
            self.hybrid_model.add_model('collaborative', self.collab_model, 
                                       weights.get('collaborative', 0.4))
            self.hybrid_model.add_model('content_based', self.content_model,
                                       weights.get('content_based', 0.25))
            self.hybrid_model.add_model('popularity', self.popularity_model,
                                       weights.get('popularity', 0.15))
            self.hybrid_model.add_model('trending', self.trending_model,
                                       weights.get('trending', 0.1))
            
            self.hybrid_model.fit()
            
            self.is_ready = True
            logger.info("Recommendation engine ready!")
            
        except Exception as e:
            logger.error(f"Failed to initialize engine: {str(e)}")
            raise
    
    def recommend_for_user(self, user_id: str, n_recommendations: int = 10,
                          category: Optional[str] = None,
                          exclude_purchased: bool = True,
                          min_rating: float = 2.0) -> List[Dict[str, Any]]:
        """
        Get recommendations for a user.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            category: Optional category filter
            exclude_purchased: Exclude already purchased products
            min_rating: Minimum product rating
        
        Returns:
            List of recommendation dictionaries with product info and explanation
        """
        if not self.is_ready:
            raise ValueError("Engine not initialized. Call initialize_models() first.")
        
        # Check cache
        cache_key = f"user_rec_{user_id}_{n_recommendations}_{category}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.debug(f"Returning cached recommendations for user {user_id}")
            return cached
        
        # Check if user is new
        if user_id not in self.users_df['user_id'].values:
            logger.info(f"New user {user_id}, using cold-start recommendations")
            return self._get_new_user_recommendations(user_id, n_recommendations, category)
        
        # Get user interactions
        user_interactions = self.interactions_df[
            self.interactions_df['user_id'] == user_id
        ]['product_id'].unique().tolist()
        
        # Get purchased products
        purchased = set()
        if exclude_purchased:
            purchased = set(
                self.interactions_df[
                    (self.interactions_df['user_id'] == user_id) &
                    (self.interactions_df['interaction_type'] == 'purchase')
                ]['product_id'].unique()
            )
        
        # Filter products
        filter_df = self.products_df.copy()
        
        if exclude_purchased:
            filter_df = filter_df[~filter_df['product_id'].isin(purchased)]
        
        filter_df = filter_df[filter_df['rating'] >= min_rating]
        filter_df = filter_df[filter_df['stock'] > 0]
        
        if category:
            filter_df = filter_df[filter_df['category'] == category]
        
        exclude_products = set(self.products_df[
            ~self.products_df['product_id'].isin(filter_df['product_id'])
        ]['product_id'].values)
        
        # Get recommendations from hybrid model
        try:
            recommendations = self.hybrid_model.predict(
                user_id=user_id,
                n_recommendations=n_recommendations * 2,
                user_interactions=user_interactions,
                exclude_products=list(exclude_products),
                products=self.products_df
            )
        except Exception as e:
            logger.warning(f"Hybrid model failed: {e}, using popularity fallback")
            recommendations = self.popularity_model.predict(user_id, n_recommendations)
        
        if not recommendations:
            logger.info(f"No recommendations found, using trending for user {user_id}")
            recommendations = self.trending_model.predict(user_id, n_recommendations)
        
        # Build response
        result = []
        for product_id, score in recommendations[:n_recommendations]:
            product = self.products_df[
                self.products_df['product_id'] == product_id
            ].iloc[0]
            
            result.append({
                'product_id': product_id,
                'product_name': product['product_name'],
                'category': product['category'],
                'price': float(product['price']),
                'rating': float(product['rating']),
                'recommendation_score': float(score),
                'explanation': self._generate_explanation(
                    user_id, product_id, user_interactions
                )
            })
        
        # Cache results
        self.cache.set(cache_key, result)
        
        return result
    
    def _get_new_user_recommendations(self, user_id: str, n_recommendations: int,
                                     category: Optional[str]) -> List[Dict[str, Any]]:
        """Get recommendations for new user."""
        recommendations = self.cold_start_model.predict(
            user_id, n_recommendations
        )
        
        if not recommendations:
            # Fallback to trending
            recommendations = self.trending_model.predict(user_id, n_recommendations)
        
        result = []
        for product_id, score in recommendations[:n_recommendations]:
            if product_id in self.products_df['product_id'].values:
                product = self.products_df[
                    self.products_df['product_id'] == product_id
                ].iloc[0]
                
                if category and product['category'] != category:
                    continue
                
                result.append({
                    'product_id': product_id,
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'price': float(product['price']),
                    'rating': float(product['rating']),
                    'recommendation_score': float(score),
                    'explanation': 'Popular among users like you'
                })
        
        return result[:n_recommendations]
    
    def similar_products(self, product_id: str, n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get similar products."""
        if not self.is_ready:
            raise ValueError("Engine not initialized")
        
        if product_id not in self.products_df['product_id'].values:
            logger.warning(f"Product {product_id} not found")
            return []
        
        recommendations = self.content_model.predict(
            user_id=product_id,
            n_recommendations=n_recommendations,
            user_interactions=[product_id]
        )
        
        result = []
        for sim_product_id, score in recommendations:
            if sim_product_id in self.products_df['product_id'].values:
                product = self.products_df[
                    self.products_df['product_id'] == sim_product_id
                ].iloc[0]
                
                result.append({
                    'product_id': sim_product_id,
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'price': float(product['price']),
                    'rating': float(product['rating']),
                    'similarity_score': float(score)
                })
        
        return result
    
    def trending_products(self, n_recommendations: int = 10) -> List[Dict[str, Any]]:
        """Get trending products."""
        if not self.is_ready:
            raise ValueError("Engine not initialized")
        
        recommendations = self.trending_model.predict(
            user_id='system',
            n_recommendations=n_recommendations
        )
        
        result = []
        for product_id, score in recommendations:
            if product_id in self.products_df['product_id'].values:
                product = self.products_df[
                    self.products_df['product_id'] == product_id
                ].iloc[0]
                
                result.append({
                    'product_id': product_id,
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'price': float(product['price']),
                    'rating': float(product['rating']),
                    'trending_score': float(score)
                })
        
        return result
    
    def frequently_bought_together(self, product_id: str, 
                                   n_recommendations: int = 5) -> List[Dict[str, Any]]:
        """Get frequently bought together products."""
        if not self.is_ready:
            raise ValueError("Engine not initialized")
        
        # Find users who bought this product
        buyers = set(
            self.interactions_df[
                (self.interactions_df['product_id'] == product_id) &
                (self.interactions_df['interaction_type'] == 'purchase')
            ]['user_id'].unique()
        )
        
        if not buyers:
            return []
        
        # Find products bought by same users
        together_purchases = self.interactions_df[
            (self.interactions_df['user_id'].isin(buyers)) &
            (self.interactions_df['interaction_type'] == 'purchase') &
            (self.interactions_df['product_id'] != product_id)
        ]['product_id'].value_counts()
        
        result = []
        for together_product_id, count in together_purchases.head(n_recommendations).items():
            if together_product_id in self.products_df['product_id'].values:
                product = self.products_df[
                    self.products_df['product_id'] == together_product_id
                ].iloc[0]
                
                result.append({
                    'product_id': together_product_id,
                    'product_name': product['product_name'],
                    'category': product['category'],
                    'price': float(product['price']),
                    'rating': float(product['rating']),
                    'co_purchase_count': int(count)
                })
        
        return result
    
    def _generate_explanation(self, user_id: str, product_id: str,
                             user_interactions: List[str]) -> str:
        """Generate human-readable explanation for recommendation."""
        if product_id in user_interactions:
            return "Similar to products you've viewed"
        
        # Check if product is trending
        trending = [p for p, _ in self.trending_model.predict('system', 20)]
        if product_id in trending:
            return "Trending right now"
        
        # Check if popular
        popular = [p for p, _ in self.popularity_model.predict('system', 20)]
        if product_id in popular:
            return "Popular in your category"
        
        return "Recommended based on your interests"
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and model information."""
        return {
            'is_ready': self.is_ready,
            'training_date': self.hybrid_model.training_date if self.hybrid_model else None,
            'models': {
                'popularity': self.popularity_model.get_model_info() if self.popularity_model else None,
                'content_based': self.content_model.get_model_info() if self.content_model else None,
                'collaborative': self.collab_model.get_model_info() if self.collab_model else None,
                'trending': self.trending_model.get_model_info() if self.trending_model else None,
                'hybrid': self.hybrid_model.get_ensemble_info() if self.hybrid_model else None
            },
            'data': {
                'n_users': len(self.users_df) if self.users_df is not None else 0,
                'n_products': len(self.products_df) if self.products_df is not None else 0,
                'n_interactions': len(self.interactions_df) if self.interactions_df is not None else 0
            },
            'cache': {
                'size': self.cache.size()
            }
        }
