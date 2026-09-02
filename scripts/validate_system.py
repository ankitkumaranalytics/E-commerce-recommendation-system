#!/usr/bin/env python
"""
System Validation Script

Validates that all components are working correctly.
"""

import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import logger
from pathlib import Path


def check_python_version():
    """Check Python version."""
    logger.info("[1/10] Checking Python version...")
    required = (3, 11)
    current = sys.version_info[:2]
    
    if current >= required:
        logger.info(f"  [OK] Python {current[0]}.{current[1]}")
        return True
    else:
        logger.error(f"  [FAIL] Python {current[0]}.{current[1]} (Need {required[0]}.{required[1]}+)")
        return False


def check_dependencies():
    """Check required dependencies."""
    logger.info("[2/10] Checking dependencies...")
    required = [
        'pandas', 'numpy', 'sklearn', 'fastapi', 'sqlalchemy',
        'yaml', 'dotenv', 'pydantic', 'joblib'
    ]
    
    missing = []
    for pkg in required:
        try:
            __import__(pkg if pkg != 'sklearn' else 'sklearn', fromlist=[''])
            logger.info(f"  [OK] {pkg}")
        except ImportError:
            logger.error(f"  [FAIL] {pkg} (MISSING)")
            missing.append(pkg)
    
    return len(missing) == 0


def check_data_files():
    """Check if data files exist."""
    logger.info("[3/10] Checking data files...")
    files = [
        'data/raw/users.csv',
        'data/raw/products.csv',
        'data/raw/interactions.csv'
    ]
    
    all_exist = True
    for file in files:
        if Path(file).exists():
            size = Path(file).stat().st_size / 1024  # KB
            logger.info(f"  [OK] {file} ({size:.1f} KB)")
        else:
            logger.error(f"  [FAIL] {file} (MISSING)")
            all_exist = False
    
    return all_exist


def check_models():
    """Check if trained models exist."""
    logger.info("[4/10] Checking trained models...")
    models = [
        'models/popularity_v1.pkl',
        'models/content_based_v1.pkl',
        'models/collaborative_v1.pkl',
        'models/trending_v1.pkl',
        'models/hybrid_v1.pkl'
    ]
    
    all_exist = True
    for model in models:
        if Path(model).exists():
            size = Path(model).stat().st_size / 1024  # KB
            logger.info(f"  [OK] {model} ({size:.1f} KB)")
        else:
            logger.error(f"  [FAIL] {model} (MISSING)")
            all_exist = False
    
    return all_exist


def check_config():
    """Check configuration file."""
    logger.info("[5/10] Checking configuration...")
    
    if Path('configs/config.yaml').exists():
        logger.info("  [OK] configs/config.yaml")
        return True
    else:
        logger.error("  [FAIL] configs/config.yaml (MISSING)")
        return False


def check_data_loading():
    """Test data loading."""
    logger.info("[6/10] Testing data loading...")
    
    try:
        from src.data import DataLoader
        users = DataLoader.load_users()
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        logger.info(f"  [OK] Loaded {len(users)} users")
        logger.info(f"  [OK] Loaded {len(products)} products")
        logger.info(f"  [OK] Loaded {len(interactions)} interactions")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] Data loading failed: {str(e)}")
        return False


def check_model_loading():
    """Test model loading."""
    logger.info("[7/10] Testing model loading...")
    
    try:
        from src.models import PopularityModel
        model = PopularityModel.load('models/popularity_v1.pkl')
        
        if model.is_fitted:
            logger.info(f"  [OK] Popularity model loaded and fitted")
            return True
        else:
            logger.error("  [FAIL] Model not fitted")
            return False
    except Exception as e:
        logger.error(f"  [FAIL] Model loading failed: {str(e)}")
        return False


def check_engine_initialization():
    """Test engine initialization."""
    logger.info("[8/10] Testing recommendation engine...")
    
    try:
        from src.data import DataLoader, DataValidator
        from src.recommendation import RecommendationEngine
        
        users = DataLoader.load_users()
        products = DataLoader.load_products()
        interactions = DataLoader.load_interactions()
        
        users = DataValidator.validate_users(users)
        products = DataValidator.validate_products(products)
        interactions = DataValidator.validate_interactions(interactions, users, products)
        
        engine = RecommendationEngine()
        engine.initialize_models(users, products, interactions)
        
        if engine.is_ready:
            logger.info(f"  [OK] Engine initialized")
            logger.info(f"    - Users: {len(users)}")
            logger.info(f"    - Products: {len(products)}")
            logger.info(f"    - Interactions: {len(interactions)}")
            return True
        else:
            logger.error("  [FAIL] Engine not ready")
            return False
    except Exception as e:
        logger.error(f"  [FAIL] Engine initialization failed: {str(e)}")
        return False


def check_api_imports():
    """Test API imports."""
    logger.info("[9/10] Testing API imports...")
    
    try:
        from src.api.main import app
        logger.info(f"  [OK] FastAPI app loaded")
        
        # Check routes
        routes = [route.path for route in app.routes]
        logger.info(f"    - Routes available: {len(routes)}")
        return True
    except Exception as e:
        logger.error(f"  [FAIL] API import failed: {str(e)}")
        return False


def check_directory_structure():
    """Check directory structure."""
    logger.info("[10/10] Checking directory structure...")
    
    required_dirs = [
        'src', 'src/api', 'src/data', 'src/models', 'src/recommendation',
        'data', 'models', 'frontend', 'tests', 'scripts', 'configs'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).is_dir():
            logger.info(f"  [OK] {dir_path}/")
        else:
            logger.error(f"  [FAIL] {dir_path}/ (MISSING)")
            all_exist = False
    
    return all_exist


def main():
    """Run all checks."""
    logger.info("="*60)
    logger.info("SYSTEM VALIDATION")
    logger.info("="*60)
    logger.info("")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files),
        ("Trained Models", check_models),
        ("Configuration", check_config),
        ("Data Loading", check_data_loading),
        ("Model Loading", check_model_loading),
        ("Engine Initialization", check_engine_initialization),
        ("API Imports", check_api_imports),
        ("Directory Structure", check_directory_structure),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
            logger.info("")
        except Exception as e:
            logger.error(f"ERROR in {name}: {str(e)}")
            logger.info("")
            results.append((name, False))
    
    # Summary
    logger.info("="*60)
    logger.info("VALIDATION SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"  {name}: {status}")
    
    logger.info("")
    logger.info(f"Result: {passed}/{total} checks passed")
    logger.info("")
    
    if passed == total:
        logger.info("SUCCESS - System ready!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. python -m src.main api")
        logger.info("2. Open browser to http://localhost:8000/docs")
        return 0
    else:
        logger.error("FAILED - Some checks did not pass")
        return 1


if __name__ == "__main__":
    sys.exit(main())
