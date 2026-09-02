"""
Common Utility Functions

Provides common utilities for the recommendation system.
"""

import hashlib
import json
from typing import Any, Dict, List
from datetime import datetime, timedelta
import numpy as np


def generate_hash(text: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    """
    Flatten nested dictionary.
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator for nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: Array of scores
        method: Normalization method (minmax or zscore)
    
    Returns:
        Normalized scores
    """
    if len(scores) == 0:
        return scores
    
    if method == "minmax":
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score == min_score:
            return np.full_like(scores, 0.5)
        return (scores - min_score) / (max_score - min_score)
    
    elif method == "zscore":
        mean = np.mean(scores)
        std = np.std(scores)
        if std == 0:
            return np.full_like(scores, 0.5)
        normalized = (scores - mean) / std
        return (normalized + 3) / 6  # Map to approximately [0, 1]
    
    return scores


def combine_scores(scores: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    """
    Combine multiple score arrays with weights.
    
    Args:
        scores: Dictionary of score arrays {name: scores}
        weights: Dictionary of weights {name: weight}
    
    Returns:
        Combined score array
    """
    combined = np.zeros(len(next(iter(scores.values()))))
    
    for name, weight in weights.items():
        if name in scores:
            combined += normalize_scores(scores[name]) * weight
    
    return combined


def get_time_decay_weight(timestamp: datetime, half_life_days: float = 30) -> float:
    """
    Calculate time decay weight for interaction.
    
    Args:
        timestamp: Interaction timestamp
        half_life_days: Half-life of interactions
    
    Returns:
        Decay weight in [0, 1]
    """
    days_ago = (datetime.now() - timestamp).days
    if days_ago < 0:
        return 1.0
    
    return 2 ** (-days_ago / half_life_days)


def remove_duplicates(items: List[str]) -> List[str]:
    """Remove duplicates preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def batch_list(items: List[Any], batch_size: int) -> List[List[Any]]:
    """Split list into batches."""
    return [items[i:i + batch_size] for i in range(0, len(items), batch_size)]


def dict_to_json(d: Dict) -> str:
    """Convert dictionary to JSON string."""
    return json.dumps(d, default=str)


def json_to_dict(s: str) -> Dict:
    """Convert JSON string to dictionary."""
    return json.loads(s)


class SimpleCache:
    """Simple in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize cache.
        
        Args:
            max_size: Maximum number of items
            ttl_seconds: Time-to-live for items
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, tuple] = {}
    
    def get(self, key: str) -> Any:
        """Get value from cache."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = min(self.cache.keys(), 
                            key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now())
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
