"""
Comprehensive Test Suite

Tests for data pipeline, models, and recommendation engine.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# Import modules
from src.data import DataLoader, DataValidator
from src.features import FeatureEngineer
from src.models import (
    PopularityModel,
    ContentBasedModel,
    CollaborativeFilteringModel,
    TrendingModel,
    HybridModel,
)
from src.models.cold_start import ColdStartModel
from src.recommendation import RecommendationEngine
from src.utils.common import normalize_scores, combine_scores, SimpleCache


# ============================================
# DATA LOADING TESTS
# ============================================

class TestDataLoading:
    """Test data loading functionality."""
    
    def test_load_users(self):
        """Test loading users dataset."""
        users = DataLoader.load_users()
        assert isinstance(users, pd.DataFrame)
        assert len(users) > 0
        assert 'user_id' in users.columns
    
    def test_load_products(self):
        """Test loading products dataset."""
        products = DataLoader.load_products()
        assert isinstance(products, pd.DataFrame)
        assert len(products) > 0
        assert 'product_id' in products.columns
    
    def test_load_interactions(self):
        """Test loading interactions dataset."""
        interactions = DataLoader.load_interactions()
        assert isinstance(interactions, pd.DataFrame)
        assert len(interactions) > 0
        assert 'user_id' in interactions.columns


# ============================================
# DATA VALIDATION TESTS
# ============================================

class TestDataValidation:
    """Test data validation functionality."""
    
    def test_validate_users(self):
        """Test user validation."""
        users = DataLoader.load_users()
        validated = DataValidator.validate_users(users)
        
        # Check age is valid
        assert (validated['age'] >= 13).all()
        assert (validated['age'] <= 120).all()
        
        # Check gender is valid
        valid_genders = ['M', 'F', 'Other']
        assert validated['gender'].isin(valid_genders).all()
    
    def test_validate_products(self):
        """Test product validation."""
        products = DataLoader.load_products()
        validated = DataValidator.validate_products(products)
        
        # Check price is positive
        assert (validated['price'] > 0).all()
        
        # Check rating is valid
        assert (validated['rating'] >= 0).all()
        assert (validated['rating'] <= 5).all()
    
    def test_validate_interactions(self):
        """Test interaction validation."""
        users = DataLoader.load_users()
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        users_clean = DataValidator.validate_users(users)
        products_clean = DataValidator.validate_products(products)
        
        validated = DataValidator.validate_interactions(
            interactions, users_clean, products_clean
        )
        
        # Check all users and products exist
        valid_user_ids = set(users_clean['user_id'].values)
        valid_product_ids = set(products_clean['product_id'].values)
        
        assert validated['user_id'].isin(valid_user_ids).all()
        assert validated['product_id'].isin(valid_product_ids).all()


# ============================================
# FEATURE ENGINEERING TESTS
# ============================================

class TestFeatureEngineering:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Load sample data."""
        users = DataLoader.load_users()
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        users = DataValidator.validate_users(users)
        products = DataValidator.validate_products(products)
        interactions = DataValidator.validate_interactions(interactions, users, products)
        
        return users, products, interactions
    
    def test_create_interaction_matrix(self, sample_data):
        """Test interaction matrix creation."""
        users, products, interactions = sample_data
        
        matrix, user_list, product_list = FeatureEngineer.create_interaction_matrix(
            interactions, users, products
        )
        
        assert isinstance(matrix, pd.DataFrame)
        assert len(user_list) > 0
        assert len(product_list) > 0
        assert matrix.shape[0] == len(user_list)
        assert matrix.shape[1] == len(product_list)


# ============================================
# MODEL TESTS
# ============================================

class TestPopularityModel:
    """Test popularity model."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create fitted popularity model."""
        users = DataLoader.load_users()
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        users = DataValidator.validate_users(users)
        products = DataValidator.validate_products(products)
        interactions = DataValidator.validate_interactions(interactions, users, products)
        
        model = PopularityModel()
        model.fit(interactions, products)
        return model
    
    def test_model_fit(self, fitted_model):
        """Test model fitting."""
        assert fitted_model.is_fitted
        assert fitted_model.product_scores is not None
        assert len(fitted_model.product_scores) > 0
    
    def test_model_predict(self, fitted_model):
        """Test model predictions."""
        predictions = fitted_model.predict('user_test', n_recommendations=5)
        
        assert isinstance(predictions, list)
        assert len(predictions) <= 5
        assert all(isinstance(p, tuple) and len(p) == 2 for p in predictions)


class TestContentBasedModel:
    """Test content-based model."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create fitted content-based model."""
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        products = DataValidator.validate_products(products)
        
        model = ContentBasedModel()
        model.fit(products, interactions)
        return model
    
    def test_model_fit(self, fitted_model):
        """Test model fitting."""
        assert fitted_model.is_fitted
        assert fitted_model.tfidf_matrix is not None
        assert fitted_model.product_ids is not None


class TestCollaborativeFilteringModel:
    """Test collaborative filtering model."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create fitted collaborative model."""
        users = DataLoader.load_users()
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        users = DataValidator.validate_users(users)
        products = DataValidator.validate_products(products)
        interactions = DataValidator.validate_interactions(interactions, users, products)
        
        matrix, user_list, product_list = FeatureEngineer.create_interaction_matrix(
            interactions, users, products
        )
        
        model = CollaborativeFilteringModel(algorithm='svd', n_factors=50)
        model.fit(matrix, user_list, product_list)
        return model
    
    def test_model_fit(self, fitted_model):
        """Test model fitting."""
        assert fitted_model.is_fitted
        assert fitted_model.user_vectors is not None
        assert fitted_model.user_ids is not None


class TestTrendingModel:
    """Test trending model."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create fitted trending model."""
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        products = DataValidator.validate_products(products)
        
        model = TrendingModel()
        model.fit(interactions, products, window_days=7)
        return model
    
    def test_model_fit(self, fitted_model):
        """Test model fitting."""
        assert fitted_model.is_fitted
        assert fitted_model.product_scores is not None


# ============================================
# UTILITY TESTS
# ============================================

class TestUtilities:
    """Test utility functions."""
    
    def test_normalize_scores(self):
        """Test score normalization."""
        scores = np.array([1, 2, 3, 4, 5])
        normalized = normalize_scores(scores, method='minmax')
        
        assert normalized.min() >= 0
        assert normalized.max() <= 1
    
    def test_simple_cache(self):
        """Test simple cache."""
        cache = SimpleCache(max_size=10, ttl_seconds=60)
        
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'
        
        cache.set('key2', 'value2')
        assert cache.size() == 2


# ============================================
# RECOMMENDATION ENGINE TESTS
# ============================================

class TestRecommendationEngine:
    """Test recommendation engine."""
    
    @pytest.fixture
    def initialized_engine(self):
        """Create initialized recommendation engine."""
        users = DataLoader.load_users()
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        users = DataValidator.validate_users(users)
        products = DataValidator.validate_products(products)
        interactions = DataValidator.validate_interactions(interactions, users, products)
        
        engine = RecommendationEngine()
        engine.initialize_models(users, products, interactions)
        return engine
    
    def test_engine_initialization(self, initialized_engine):
        """Test engine initialization."""
        assert initialized_engine.is_ready
        assert initialized_engine.popularity_model is not None
        assert initialized_engine.content_model is not None
        assert initialized_engine.collab_model is not None
    
    def test_recommend_for_user(self, initialized_engine):
        """Test user recommendations."""
        user_id = 'U00001'
        recommendations = initialized_engine.recommend_for_user(user_id, n_recommendations=5)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 5
        assert all('product_id' in r and 'recommendation_score' in r for r in recommendations)
    
    def test_similar_products(self, initialized_engine):
        """Test similar products."""
        product_id = 'P00001'
        similar = initialized_engine.similar_products(product_id, n_recommendations=5)
        
        assert isinstance(similar, list)
    
    def test_trending_products(self, initialized_engine):
        """Test trending products."""
        trending = initialized_engine.trending_products(n_recommendations=10)
        
        assert isinstance(trending, list)
    
    def test_engine_status(self, initialized_engine):
        """Test engine status."""
        status = initialized_engine.get_engine_status()
        
        assert status['is_ready']
        assert 'data' in status
        assert status['data']['n_users'] > 0


# ============================================
# EDGE CASES
# ============================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_cold_start_new_user(self):
        """Test cold-start for new user."""
        users = DataLoader.load_users()
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        users = DataValidator.validate_users(users)
        products = DataValidator.validate_products(products)
        interactions = DataValidator.validate_interactions(interactions, users, products)
        
        model = ColdStartModel()
        model.fit(interactions, products)
        
        # New user
        recommendations = model.predict('new_user_12345', n_recommendations=5)
        assert len(recommendations) > 0
    
    def test_empty_user_interactions(self):
        """Test with user having no interactions."""
        users = DataLoader.load_users()
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        users = DataValidator.validate_users(users)
        products = DataValidator.validate_products(products)
        interactions = DataValidator.validate_interactions(interactions, users, products)
        
        model = PopularityModel()
        model.fit(interactions, products)
        
        # Predict for user with no interactions
        predictions = model.predict('no_interaction_user', n_recommendations=5)
        assert isinstance(predictions, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
