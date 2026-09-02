"""
Data Loading Module

Handles loading and initial validation of datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from src.utils.logger import logger


class DataLoader:
    """Load datasets from CSV files."""
    
    @staticmethod
    def load_users(path: str = "data/raw/users.csv") -> pd.DataFrame:
        """Load users dataset."""
        logger.info(f"Loading users from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} users with columns: {list(df.columns)}")
        return df
    
    @staticmethod
    def load_products(path: str = "data/raw/products.csv") -> pd.DataFrame:
        """Load products dataset."""
        logger.info(f"Loading products from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} products with columns: {list(df.columns)}")
        return df
    
    @staticmethod
    def load_interactions(path: str = "data/raw/interactions.csv") -> pd.DataFrame:
        """Load interactions dataset."""
        logger.info(f"Loading interactions from {path}")
        df = pd.read_csv(path)
        logger.info(f"Loaded {len(df)} interactions with columns: {list(df.columns)}")
        return df
    
    @staticmethod
    def load_all(
        users_path: str = "data/raw/users.csv",
        products_path: str = "data/raw/products.csv",
        interactions_path: str = "data/raw/interactions.csv"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load all datasets."""
        users = DataLoader.load_users(users_path)
        products = DataLoader.load_products(products_path)
        interactions = DataLoader.load_interactions(interactions_path)
        return users, products, interactions


class DataValidator:
    """Validate dataset integrity."""
    
    @staticmethod
    def validate_users(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate users dataset.
        
        Returns:
            Cleaned dataset
        """
        logger.info("Validating users dataset")
        
        required_columns = ['user_id', 'age', 'gender', 'location', 'signup_date']
        assert all(col in df.columns for col in required_columns), \
            f"Missing required columns. Got: {list(df.columns)}"
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['user_id'])
        
        # Validate age
        df = df[(df['age'] >= 13) & (df['age'] <= 120)]
        
        # Validate gender
        valid_genders = ['M', 'F', 'Other']
        df = df[df['gender'].isin(valid_genders)]
        
        # Validate dates
        df['signup_date'] = pd.to_datetime(df['signup_date'], errors='coerce')
        df = df.dropna(subset=['signup_date'])
        
        removed = initial_rows - len(df)
        logger.info(f"Removed {removed} invalid user records. Remaining: {len(df)}")
        
        return df
    
    @staticmethod
    def validate_products(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate products dataset.
        
        Returns:
            Cleaned dataset
        """
        logger.info("Validating products dataset")
        
        required_columns = ['product_id', 'product_name', 'category', 'price', 'rating', 'stock']
        assert all(col in df.columns for col in required_columns), \
            f"Missing required columns. Got: {list(df.columns)}"
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['product_id'])
        
        # Validate price
        df = df[df['price'] > 0]
        
        # Validate rating
        df = df[(df['rating'] >= 0) & (df['rating'] <= 5)]
        
        # Validate stock
        df = df[df['stock'] >= 0]
        
        # Validate date
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        
        removed = initial_rows - len(df)
        logger.info(f"Removed {removed} invalid product records. Remaining: {len(df)}")
        
        return df
    
    @staticmethod
    def validate_interactions(df: pd.DataFrame, 
                            valid_users: pd.DataFrame,
                            valid_products: pd.DataFrame) -> pd.DataFrame:
        """
        Validate interactions dataset.
        
        Returns:
            Cleaned dataset
        """
        logger.info("Validating interactions dataset")
        
        required_columns = ['interaction_id', 'user_id', 'product_id', 'interaction_type', 'timestamp']
        assert all(col in df.columns for col in required_columns), \
            f"Missing required columns. Got: {list(df.columns)}"
        
        initial_rows = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['interaction_id'])
        
        # Keep only valid users and products
        valid_user_ids = set(valid_users['user_id'].values)
        valid_product_ids = set(valid_products['product_id'].values)
        
        df = df[df['user_id'].isin(valid_user_ids)]
        df = df[df['product_id'].isin(valid_product_ids)]
        
        # Validate interaction types
        valid_types = {'view', 'click', 'wishlist', 'cart', 'purchase'}
        df = df[df['interaction_type'].isin(valid_types)]
        
        # Validate timestamp
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        removed = initial_rows - len(df)
        logger.info(f"Removed {removed} invalid interaction records. Remaining: {len(df)}")
        
        return df
    
    @staticmethod
    def validate_all(users: pd.DataFrame, products: pd.DataFrame, 
                    interactions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Validate all datasets together."""
        logger.info("Starting validation pipeline")
        
        users = DataValidator.validate_users(users)
        products = DataValidator.validate_products(products)
        interactions = DataValidator.validate_interactions(interactions, users, products)
        
        logger.info("Validation complete")
        
        return users, products, interactions
