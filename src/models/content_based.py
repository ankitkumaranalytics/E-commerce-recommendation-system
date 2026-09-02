"""
Content-Based Recommendation Model

Uses product metadata to find similar products.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.models.base import BaseRecommender
from src.utils.logger import logger


class ContentBasedModel(BaseRecommender):
    """
    Content-based recommender.
    
    Recommends products similar to products the user has interacted with.
    Uses TF-IDF vectorization of product descriptions and metadata.
    """
    
    def __init__(self, name: str = "content_based"):
        """Initialize content-based model."""
        super().__init__(name)
        self.vectorizer: TfidfVectorizer = None
        self.tfidf_matrix: np.ndarray = None
        self.product_ids: List[str] = None
        self.product_idx_map: Dict[str, int] = None
    
    def fit(self, products: pd.DataFrame, interactions: pd.DataFrame,
            max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)) -> None:
        """
        Train content-based model.
        
        Args:
            products: Products dataframe
            interactions: Interactions dataframe (for statistics)
            max_features: Maximum TF-IDF features
            ngram_range: N-gram range for TF-IDF
        """
        logger.info(f"Training {self.name} model")
        
        # Combine text features
        products['content'] = (
            products['product_name'].fillna('') + ' ' +
            products['category'].fillna('') + ' ' +
            products['subcategory'].fillna('') + ' ' +
            products['brand'].fillna('') + ' ' +
            products['description'].fillna('')
        )
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            min_df=1,
            max_df=0.95
        )
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(products['content']).toarray()
        
        # Store product IDs for indexing
        self.product_ids = products['product_id'].tolist()
        self.product_idx_map = {pid: idx for idx, pid in enumerate(self.product_ids)}
        
        # Compute similarity matrix
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        self.is_fitted = True
        self.training_date = datetime.now()
        self.metadata = {
            'max_features': max_features,
            'ngram_range': ngram_range,
            'n_products': len(self.product_ids),
            'feature_names': self.vectorizer.get_feature_names_out().tolist()[:10]  # Top 10
        }
        
        logger.info(f"Training complete. TF-IDF matrix shape: {self.tfidf_matrix.shape}")
    
    def predict(self, user_id: str, n_recommendations: int = 10,
                user_interactions: List[str] = None, exclude_products: List[str] = None,
                **kwargs) -> List[Tuple[str, float]]:
        """
        Get content-based recommendations.
        
        Args:
            user_id: User ID
            n_recommendations: Number of recommendations
            user_interactions: List of products user has interacted with
            exclude_products: Products to exclude from recommendations
            **kwargs: Additional arguments
        
        Returns:
            List of (product_id, score) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call fit() first.")
        
        if user_interactions is None or len(user_interactions) == 0:
            logger.debug(f"No interactions for user {user_id}, using zero scores")
            return []
        
        # Get indices of user's interactions
        interaction_indices = [
            self.product_idx_map[pid] for pid in user_interactions
            if pid in self.product_idx_map
        ]
        
        if not interaction_indices:
            return []
        
        # Average similarity to all interacted products
        similarities = np.mean(self.similarity_matrix[interaction_indices], axis=0)
        
        # Get product scores
        product_scores = [
            (self.product_ids[i], float(similarities[i]))
            for i in range(len(self.product_ids))
        ]
        
        # Filter out user interactions and excluded products
        exclude_set = set(user_interactions or [])
        if exclude_products:
            exclude_set.update(exclude_products)
        
        product_scores = [
            (pid, score) for pid, score in product_scores
            if pid not in exclude_set and score > 0
        ]
        
        # Sort by score
        product_scores.sort(key=lambda x: x[1], reverse=True)
        
        return product_scores[:n_recommendations]
