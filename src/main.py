"""
Application Entry Point

Provides main entry point for the application with multiple modes:
- train: Train recommendation models
- api: Start FastAPI server
- dev: Development mode
"""

import sys
import argparse
import os
from src.utils.logger import logger


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="E-Commerce Recommendation System"
    )
    
    parser.add_argument(
        "mode",
        nargs="?",
        default="api",
        choices=["train", "api", "dev"],
        help="Execution mode"
    )
    
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="API host (for api mode)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", os.getenv("API_PORT", "8000"))),
        help="API port (for api mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        logger.info("Starting model training")
        from scripts.train_models import train_recommendation_system
        train_recommendation_system()
    
    elif args.mode == "api" or args.mode == "dev":
        logger.info(f"Starting API server in {args.mode} mode")
        import uvicorn
        reload = args.mode == "dev"
        
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=reload,
            log_level="info"
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
