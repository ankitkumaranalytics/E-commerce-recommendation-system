"""
Training Script

Orchestrates the complete data and model training pipeline.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataLoader, DataValidator
from src.features import FeatureEngineer
from src.recommendation import RecommendationEngine
from src.utils.logger import logger
from pathlib import Path
import pickle


def train_recommendation_system():
    """
    Train the complete recommendation system.
    
    Steps:
    1. Load raw data
    2. Validate data
    3. Save processed data
    4. Initialize and train recommendation engine
    5. Save models
    """
    
    logger.info("="*60)
    logger.info("STARTING RECOMMENDATION SYSTEM TRAINING")
    logger.info("="*60)
    
    try:
        # Step 1: Load data
        logger.info("\n[STEP 1] Loading raw data...")
        users, products, interactions = DataLoader.load_all()
        logger.info(f"Loaded {len(users)} users, {len(products)} products, {len(interactions)} interactions")
        
        # Step 2: Validate data
        logger.info("\n[STEP 2] Validating and cleaning data...")
        users_clean, products_clean, interactions_clean = DataValidator.validate_all(
            users, products, interactions
        )
        logger.info(f"After validation: {len(users_clean)} users, {len(products_clean)} products, {len(interactions_clean)} interactions")
        
        # Step 3: Save processed data
        logger.info("\n[STEP 3] Saving processed data...")
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        
        users_clean.to_csv("data/processed/users.csv", index=False)
        products_clean.to_csv("data/processed/products.csv", index=False)
        interactions_clean.to_csv("data/processed/interactions.csv", index=False)
        logger.info("Processed data saved")
        
        # Step 4: Initialize recommendation engine
        logger.info("\n[STEP 4] Initializing recommendation engine...")
        engine = RecommendationEngine()
        
        engine.initialize_models(
            users_clean,
            products_clean,
            interactions_clean,
            weights={
                'collaborative': 0.40,
                'content_based': 0.25,
                'popularity': 0.15,
                'trending': 0.10,
                'business_rules': 0.10
            }
        )
        
        # Step 5: Save engine and models
        logger.info("\n[STEP 5] Saving trained engine and models...")
        Path("models").mkdir(parents=True, exist_ok=True)
        
        engine.popularity_model.save("models/popularity_v1.pkl")
        engine.content_model.save("models/content_based_v1.pkl")
        engine.collab_model.save("models/collaborative_v1.pkl")
        engine.trending_model.save("models/trending_v1.pkl")
        engine.hybrid_model.save("models/hybrid_v1.pkl")
        
        # Save engine metadata
        import json
        metadata = {
            'training_date': engine.hybrid_model.training_date.isoformat(),
            'data_stats': {
                'n_users': len(users_clean),
                'n_products': len(products_clean),
                'n_interactions': len(interactions_clean)
            },
            'models': {
                'popularity': engine.popularity_model.get_model_info(),
                'content_based': engine.content_model.get_model_info(),
                'collaborative': engine.collab_model.get_model_info(),
                'trending': engine.trending_model.get_model_info(),
                'hybrid': engine.hybrid_model.get_ensemble_info()
            }
        }
        
        with open("models/metadata_v1.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Models saved to models/ directory")
        
        # Step 6: Test engine
        logger.info("\n[STEP 6] Testing engine on sample users...")
        test_users = users_clean.head(5)['user_id'].tolist()
        
        for user_id in test_users:
            try:
                recs = engine.recommend_for_user(user_id, n_recommendations=5)
                logger.info(f"User {user_id}: Got {len(recs)} recommendations")
            except Exception as e:
                logger.warning(f"Error for user {user_id}: {e}")
        
        # Print engine status
        logger.info("\n[STEP 7] Engine Status:")
        status = engine.get_engine_status()
        logger.info(f"  - Is Ready: {status['is_ready']}")
        logger.info(f"  - Users: {status['data']['n_users']}")
        logger.info(f"  - Products: {status['data']['n_products']}")
        logger.info(f"  - Interactions: {status['data']['n_interactions']}")
        
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE - ENGINE READY FOR USE")
        logger.info("="*60)
        
        return engine
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    train_recommendation_system()
