"""
Database Models

SQLAlchemy ORM models for the application.
"""

from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    Text, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User model."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), unique=True, nullable=False, index=True)
    age = Column(Integer)
    gender = Column(String(10))
    location = Column(String(100))
    email = Column(String(255), unique=True, index=True)
    password_hash = Column(String(255))
    role = Column(String(20), default='CUSTOMER', nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    email_verified = Column(Boolean, default=False, nullable=False)
    phone = Column(String(30))
    signup_date = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    interactions = relationship('Interaction', back_populates='user')
    recommendations = relationship('Recommendation', back_populates='user')
    
    __table_args__ = (
        Index('ix_users_user_id_created_at', 'user_id', 'created_at'),
    )


class Product(Base):
    """Product model."""
    __tablename__ = 'products'
    
    id = Column(Integer, primary_key=True)
    product_id = Column(String(50), unique=True, nullable=False, index=True)
    product_name = Column(String(255), nullable=False)
    category = Column(String(100), index=True)
    subcategory = Column(String(100))
    brand = Column(String(100))
    description = Column(Text)
    price = Column(Float, nullable=False)
    rating = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    stock = Column(Integer, default=0)
    image_url = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    interactions = relationship('Interaction', back_populates='product')
    recommendations = relationship('Recommendation', back_populates='product')
    
    __table_args__ = (
        Index('ix_products_category_rating', 'category', 'rating'),
        Index('ix_products_product_id_created_at', 'product_id', 'created_at'),
    )


class Interaction(Base):
    """User-product interaction model."""
    __tablename__ = 'interactions'
    
    id = Column(Integer, primary_key=True)
    interaction_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(50), ForeignKey('users.user_id'), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey('products.product_id'), nullable=False, index=True)
    interaction_type = Column(String(20), nullable=False)  # view, click, wishlist, cart, purchase
    weight = Column(Float, default=1.0)  # Computed weight based on type
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='interactions')
    product = relationship('Product', back_populates='interactions')
    
    __table_args__ = (
        Index('ix_interactions_user_product', 'user_id', 'product_id'),
        Index('ix_interactions_type_timestamp', 'interaction_type', 'timestamp'),
    )


class Recommendation(Base):
    """Recommendation record model."""
    __tablename__ = 'recommendations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), ForeignKey('users.user_id'), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey('products.product_id'), nullable=False, index=True)
    recommendation_score = Column(Float, nullable=False)
    explanation = Column(String(255))
    model_version = Column(String(20), default='v1')
    category_filter = Column(String(100))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    clicked = Column(Boolean, default=False)
    purchased = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    user = relationship('User', back_populates='recommendations')
    product = relationship('Product', back_populates='recommendations')
    
    __table_args__ = (
        Index('ix_recommendations_user_timestamp', 'user_id', 'timestamp'),
        Index('ix_recommendations_product_timestamp', 'product_id', 'timestamp'),
    )


class ModelVersion(Base):
    """Model version tracking."""
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True)
    version = Column(String(20), unique=True, nullable=False)
    model_type = Column(String(50), nullable=False)  # popularity, content, collab, trending, hybrid
    training_date = Column(DateTime, nullable=False)
    data_stats = Column(Text)  # JSON string with training stats
    metrics = Column(Text)  # JSON string with evaluation metrics
    model_path = Column(String(255))
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_model_versions_type_created_at', 'model_type', 'created_at'),
    )


class Event(Base):
    """Event tracking model."""
    __tablename__ = 'events'
    
    id = Column(Integer, primary_key=True)
    event_type = Column(String(50), nullable=False, index=True)
    user_id = Column(String(50), index=True)
    product_id = Column(String(50), index=True)
    session_id = Column(String(100))
    data = Column(Text)  # JSON string with event data
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_events_type_timestamp', 'event_type', 'timestamp'),
    )


class RecommendationMetrics(Base):
    """Aggregated recommendation metrics."""
    __tablename__ = 'recommendation_metrics'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, default=datetime.utcnow, index=True)
    total_recommendations = Column(Integer, default=0)
    total_clicks = Column(Integer, default=0)
    total_purchases = Column(Integer, default=0)
    unique_users = Column(Integer, default=0)
    avg_score = Column(Float, default=0.0)
    model_version = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)


class Store(Base):
    """Seller storefront."""
    __tablename__ = "stores"
    id = Column(Integer, primary_key=True)
    store_id = Column(String(50), unique=True, nullable=False, index=True)
    seller_user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False, index=True)
    name = Column(String(150), nullable=False)
    description = Column(Text)
    logo_url = Column(String(500))
    is_verified = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Address(Base):
    """Customer delivery address."""
    __tablename__ = "addresses"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False, index=True)
    label = Column(String(50), nullable=False)
    recipient_name = Column(String(150), nullable=False)
    phone = Column(String(30), nullable=False)
    line1 = Column(String(255), nullable=False)
    line2 = Column(String(255))
    city = Column(String(100), nullable=False)
    state = Column(String(100), nullable=False)
    postal_code = Column(String(20), nullable=False)
    country = Column(String(80), default="India", nullable=False)
    is_default = Column(Boolean, default=False, nullable=False)


class Cart(Base):
    """Persistent one-cart-per-user container."""
    __tablename__ = "carts"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), unique=True, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    items = relationship("CartItem", back_populates="cart", cascade="all, delete-orphan")


class CartItem(Base):
    """Cart line item with a snapshot of the selected quantity."""
    __tablename__ = "cart_items"
    id = Column(Integer, primary_key=True)
    cart_id = Column(Integer, ForeignKey("carts.id"), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False)
    quantity = Column(Integer, nullable=False, default=1)
    cart = relationship("Cart", back_populates="items")
    __table_args__ = (UniqueConstraint("cart_id", "product_id", name="uq_cart_product"),)


class WishlistItem(Base):
    """Saved product for a customer."""
    __tablename__ = "wishlist_items"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("user_id", "product_id", name="uq_wishlist_product"),)


class Order(Base):
    """Order aggregate and lifecycle state."""
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True)
    order_id = Column(String(50), unique=True, nullable=False, index=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False, index=True)
    status = Column(String(30), default="PLACED", nullable=False, index=True)
    payment_status = Column(String(30), default="PENDING", nullable=False)
    total_amount = Column(Float, nullable=False)
    address_snapshot = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    items = relationship("OrderItem", back_populates="order", cascade="all, delete-orphan")


class OrderItem(Base):
    """Immutable order line item."""
    __tablename__ = "order_items"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False)
    product_name = Column(String(255), nullable=False)
    unit_price = Column(Float, nullable=False)
    quantity = Column(Integer, nullable=False)
    order = relationship("Order", back_populates="items")


class Payment(Base):
    """Payment state without storing card data."""
    __tablename__ = "payments"
    id = Column(Integer, primary_key=True)
    order_id = Column(Integer, ForeignKey("orders.id"), unique=True, nullable=False)
    provider = Column(String(40), nullable=False)
    provider_reference = Column(String(255))
    status = Column(String(30), default="PENDING", nullable=False)
    amount = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class Review(Base):
    """Verified-purchase product review."""
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False, index=True)
    rating = Column(Integer, nullable=False)
    body = Column(Text, nullable=False)
    is_approved = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("user_id", "product_id", name="uq_review_product"),)


class Notification(Base):
    """In-app notification, extensible to email/push later."""
    __tablename__ = "notifications"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(50), ForeignKey("users.user_id"), nullable=False, index=True)
    notification_type = Column(String(50), nullable=False)
    title = Column(String(150), nullable=False)
    message = Column(Text, nullable=False)
    is_read = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
