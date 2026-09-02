"""
Feature Engineering Module

Creates features from raw data for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from src.utils.logger import logger


class FeatureEngineer:
    """Create features from raw data."""
    
    @staticmethod
    def create_user_features(users: pd.DataFrame) -> pd.DataFrame:
        """
        Create user features.
        
        Args:
            users: User dataframe
        
        Returns:
            User features dataframe
        """
        logger.info("Creating user features")
        
        features = users.copy()
        
        # Calculate days since signup
        features['days_since_signup'] = (
            pd.Timestamp.now() - features['signup_date']
        ).dt.days
        
        # Age group binning
        features['age_group'] = pd.cut(features['age'], 
                                      bins=[0, 25, 35, 50, 100],
                                      labels=['18-25', '26-35', '36-50', '50+'])
        
        return features
    
    @staticmethod
    def create_product_features(products: pd.DataFrame) -> pd.DataFrame:
        """
        Create product features.
        
        Args:
            products: Product dataframe
        
        Returns:
            Product features dataframe
        """
        logger.info("Creating product features")
        
        features = products.copy()
        
        # Price buckets
        features['price_bucket'] = pd.qcut(features['price'], 
                                          q=4, 
                                          labels=['Budget', 'Mid', 'Premium', 'Luxury'],
                                          duplicates='drop')
        
        # Rating categories
        features['rating_category'] = pd.cut(features['rating'],
                                            bins=[0, 2, 3, 4, 5],
                                            labels=['Poor', 'Fair', 'Good', 'Excellent'])
        
        # Stock availability
        features['is_in_stock'] = (features['stock'] > 0).astype(int)
        
        # Days since product creation
        features['days_since_created'] = (
            pd.Timestamp.now() - features['created_at']
        ).dt.days
        
        return features
    
    @staticmethod
    def create_interaction_matrix(
        interactions: pd.DataFrame,
        users: pd.DataFrame,
        products: pd.DataFrame,
        interaction_weights: Dict[str, int] = None
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Create weighted user-item interaction matrix.
        
        Args:
            interactions: Interactions dataframe
            users: Users dataframe
            products: Products dataframe
            interaction_weights: Weights for each interaction type
        
        Returns:
            Interaction matrix, user list, product list
        """
        logger.info("Creating interaction matrix")
        
        if interaction_weights is None:
            interaction_weights = {
                'view': 1,
                'click': 2,
                'wishlist': 3,
                'cart': 4,
                'purchase': 5
            }
        
        # Apply weights
        interactions_weighted = interactions.copy()
        interactions_weighted['weight'] = interactions_weighted['interaction_type'].map(
            interaction_weights
        )
        
        # Create matrix
        matrix = interactions_weighted.pivot_table(
            index='user_id',
            columns='product_id',
            values='weight',
            aggfunc='sum',
            fill_value=0
        )
        
        user_list = matrix.index.tolist()
        product_list = matrix.columns.tolist()
        
        logger.info(f"Created interaction matrix: {matrix.shape}")
        
        return matrix, user_list, product_list
    
    @staticmethod
    def create_temporal_features(interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from interactions.
        
        Args:
            interactions: Interactions dataframe
        
        Returns:
            Interactions with temporal features
        """
        logger.info("Creating temporal features")
        
        features = interactions.copy()
        
        # Extract time components
        features['hour'] = features['timestamp'].dt.hour
        features['day_of_week'] = features['timestamp'].dt.dayofweek
        features['days_ago'] = (pd.Timestamp.now() - features['timestamp']).dt.days
        
        return features
    
    @staticmethod
    def create_statistical_features(
        interactions: pd.DataFrame,
        users: pd.DataFrame,
        products: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create statistical features at user and product levels.
        
        Args:
            interactions: Interactions dataframe
            users: Users dataframe
            products: Products dataframe
        
        Returns:
            User stats, product stats
        """
        logger.info("Creating statistical features")
        
        # User stats
        user_stats = interactions.groupby('user_id').agg({
            'interaction_id': 'count',
            'product_id': 'nunique'
        }).rename(columns={
            'interaction_id': 'total_interactions',
            'product_id': 'unique_products'
        })
        
        user_stats['avg_interactions'] = (
            user_stats['total_interactions'] / 
            (user_stats['unique_products'] + 1)
        )
        
        # Product stats
        product_stats = interactions.groupby('product_id').agg({
            'interaction_id': 'count',
            'user_id': 'nunique'
        }).rename(columns={
            'interaction_id': 'total_interactions',
            'user_id': 'unique_users'
        })
        
        product_stats['avg_users'] = (
            product_stats['total_interactions'] / 
            (product_stats['unique_users'] + 1)
        )
        
        logger.info(f"User stats: {user_stats.shape}, Product stats: {product_stats.shape}")
        
        return user_stats, product_stats
    
    @staticmethod
    def scale_features(features: pd.DataFrame, 
                      method: str = 'minmax',
                      numeric_cols: List[str] = None) -> Tuple[pd.DataFrame, object]:
        """
        Scale numeric features.
        
        Args:
            features: Features dataframe
            method: Scaling method (minmax or standard)
            numeric_cols: List of numeric columns to scale
        
        Returns:
            Scaled features, scaler object
        """
        logger.info(f"Scaling features using {method}")
        
        if numeric_cols is None:
            numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        
        features_scaled = features.copy()
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        
        features_scaled[numeric_cols] = scaler.fit_transform(
            features[numeric_cols]
        )
        
        return features_scaled, scaler
