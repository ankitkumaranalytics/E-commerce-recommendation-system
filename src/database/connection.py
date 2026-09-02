"""
Database Initialization and Connection

Manages database connections and initialization.
"""

import os
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from src.database.models import Base
from src.utils.logger import logger
from typing import Optional


class DatabaseConnection:
    """Database connection manager."""
    
    def __init__(self):
        """Initialize database connection."""
        self.engine = None
        self.SessionLocal = None
    
    def initialize(self, use_sqlite: bool = True, sqlite_path: str = "data/ecommerce_rec.db"):
        """
        Initialize database connection.
        
        Args:
            use_sqlite: Use SQLite instead of PostgreSQL
            sqlite_path: Path for SQLite database
        """
        try:
            if use_sqlite:
                logger.info(f"Initializing SQLite database at {sqlite_path}")
                # Create directory if needed
                os.makedirs(os.path.dirname(sqlite_path) if os.path.dirname(sqlite_path) else '.', exist_ok=True)
                
                # SQLite connection string
                database_url = f"sqlite:///{sqlite_path}"
                
                # Use StaticPool for SQLite to avoid issues with multiple threads
                self.engine = create_engine(
                    database_url,
                    connect_args={"check_same_thread": False},
                    poolclass=StaticPool,
                    echo=False
                )
            else:
                # PostgreSQL connection
                db_host = os.getenv("DB_HOST", "localhost")
                db_port = os.getenv("DB_PORT", "5432")
                db_name = os.getenv("DB_NAME", "ecommerce_rec")
                db_user = os.getenv("DB_USER", "postgres")
                db_password = os.getenv("DB_PASSWORD", "")
                
                logger.info(f"Initializing PostgreSQL database at {db_host}:{db_port}/{db_name}")
                
                database_url = (
                    f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
                )
                
                self.engine = create_engine(
                    database_url,
                    pool_size=10,
                    max_overflow=20,
                    echo=False
                )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create tables
            logger.info("Creating database tables")
            Base.metadata.create_all(bind=self.engine)
            self._migrate_user_auth_columns()
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise

    def _migrate_user_auth_columns(self):
        """Add backwards-compatible auth columns to existing development DBs."""
        if self.engine is None:
            return
        columns = {column["name"] for column in inspect(self.engine).get_columns("users")}
        additions = {
            "email": "VARCHAR(255)",
            "password_hash": "VARCHAR(255)",
            "role": "VARCHAR(20) DEFAULT 'CUSTOMER'",
            "is_active": "BOOLEAN DEFAULT 1",
            "email_verified": "BOOLEAN DEFAULT 0",
            "phone": "VARCHAR(30)",
        }
        with self.engine.begin() as connection:
            for name, definition in additions.items():
                if name not in columns:
                    connection.execute(text(f"ALTER TABLE users ADD COLUMN {name} {definition}"))
    
    def get_session(self) -> Session:
        """Get database session."""
        if self.SessionLocal is None:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        return self.SessionLocal()
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Global database instance
db = DatabaseConnection()


def get_db() -> Session:
    """Dependency for FastAPI to get database session."""
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()
