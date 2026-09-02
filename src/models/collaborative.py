"""
Collaborative Filtering Recommendation Model

Uses user-item interactions to find similar users or items.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, vstack
from src.models.base import BaseRecommender
from src.utils.logger import logger


class CollaborativeFilteringModel(BaseRecommender):
    """
    Collaborative filtering recommender using matrix factorization.
    
    Implements user-item interactions using SVD and cosine similarity.
    """
    
    def __init__(self, name: str = "collaborative", algorithm: str = "svd", n_factors: int = 50):
        """
        Initialize collaborative filtering model.
        
        Args:
            name: Model name
            algorithm: Algorithm (svd or knn)
            n_factors: Number of latent factors for SVD
        """
        super().__init__(name)
        self.algorithm = algorithm
        self.n_factors = n_factors
        self.interaction_matrix: Optional[csr_matrix] = None
        self.user_ids: List[str] = None
        self.product_ids: List[str] = None
        self.user_idx_map: Dict[str, int] = None
        self.product_idx_map: Dict[str, int] = None
        self.user_vectors: Optional[np.ndarray] = None
        self.item_vectors: Optional[np.ndarray] = None
        self.item_similarity: Optional[np.ndarray] = None
    
    def fit(self, interaction_matrix: pd.DataFrame,
            user_list: List[str], product_list: List[str]) -> None:
        """
        Train collaborative filtering model.
        
        Args:
            interaction_matrix: User-item interaction matrix (user_id x product_id)
            user_list: List of user IDs
            product_list: List of product IDs
        """
        logger.info(f"Training {self.name} model with {self.algorithm} algorithm")
        
        # Store IDs
        self.user_ids = user_list
        self.product_ids = product_list
        self.user_idx_map = {uid: idx for idx, uid in enumerate(user_list)}
        self.product_idx_map = {pid: idx for idx, pid in enumerate(product_list)}
        
        # Convert to sparse matrix
        self.interaction_matrix = csr_matrix(interaction_matrix.values)
        
        if self.algorithm == "svd":
            self._fit_svd()
        elif self.algorithm == "knn":
            self._fit_knn()
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        self.is_fitted = True
        self.training_date = datetime.now()
        self.metadata = {
            'algorithm': self.algorithm,
            'n_factors': self.n_factors,
            'n_users': len(self.user_ids),
            'n_products': len(self.product_ids),
            'sparsity': 1 - (self.interaction_matrix.nnz / 
                           (len(self.user_ids) * len(self.product_ids)))
        }
        
        logger.info(f"Training complete. Matrix sparsity: {self.metadata['sparsity']:.2%}")
    
    def _fit_svd(self) -> None:
        """Fit SVD model."""
        logger.debug("Applying SVD factorization")
        
        svd = TruncatedSVD(n_components=min(self.n_factors, 
                                           min(self.interaction_matrix.shape) - 1))
        user_vectors = svd.fit_transform(self.interaction_matrix)
        
        # Item vectors from components
        item_vectors = svd.components_.T
        
        self.user_vectors = user_vectors
        self.item_vectors = item_vectors
        
        # Compute item-item similarity
        self.item_similarity = cosine_similarity(item_vectors)
    
    def _fit_knn(self) -> None:
        """Fit KNN-based model."""
        logger.debug("Computing item-item similarity matrix")
        
        # Compute item-item similarity using cosine
        self.item_similarity = cosine_similarity(self.interaction_matrix.T)
        
        # For user vectors, use interaction matrix directly
        self.user_vectors = self.interaction_matrix.toarray()
    
    def predict(self, user_id: str, n_recommendations: int = 10,
                user_interactions: List[str] = None, exclude_products: List[str] = None,
                **kwargs) -> List[Tuple[str, float]]:
        """
        Get collaborative filtering recommendations.
        
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
            logger.debug(f"No interactions for user {user_id}")
            return []
        
        # Get indices of user's interactions
        interaction_indices = [
            self.product_idx_map[pid] for pid in user_interactions
            if pid in self.product_idx_map
        ]
        
        if not interaction_indices:
            return []
        
        # Compute scores based on item-item similarity
        scores = np.zeros(len(self.product_ids))
        
        for idx in interaction_indices:
            scores += self.item_similarity[idx]
        
        # Normalize by number of interactions
        scores = scores / (len(interaction_indices) + 1e-10)
        
        # Get product scores
        product_scores = [
            (self.product_ids[i], float(scores[i]))
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
