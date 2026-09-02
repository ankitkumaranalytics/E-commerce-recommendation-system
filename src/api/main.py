"""
FastAPI Application

Main API server for the recommendation system.
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import os
from datetime import datetime
import pandas as pd
from src.utils.logger import logger
from src.database import db, get_db
from src.database.models import (
    User, Product, Interaction, Recommendation, Event, Cart, CartItem, WishlistItem
)
from src.recommendation import RecommendationEngine
from sqlalchemy.orm import Session
import json
import re
from src.api.auth import create_access_token, get_current_user, hash_password, verify_password


# ============================================
# PYDANTIC SCHEMAS
# ============================================

class ProductSchema(BaseModel):
    """Product schema."""
    product_id: str
    product_name: str
    category: str
    price: float
    rating: float
    stock: int
    
    class Config:
        from_attributes = True


class UserSchema(BaseModel):
    """User schema."""
    user_id: str
    age: int
    gender: str
    location: str
    
    class Config:
        from_attributes = True


class RegisterRequest(BaseModel):
    user_id: str
    email: str
    password: str
    age: int = Field(18, ge=13, le=120)
    gender: str = "unspecified"
    location: str = "unknown"
    role: str = "CUSTOMER"


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    role: str


class CartItemRequest(BaseModel):
    product_id: str
    quantity: int = Field(1, ge=1, le=100)


class CartItemResponse(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    unit_price: float
    subtotal: float


class CartResponse(BaseModel):
    items: List[CartItemResponse]
    total: float


class RecommendationResponse(BaseModel):
    """Recommendation response schema."""
    product_id: str
    product_name: str
    category: str
    price: float
    rating: float
    recommendation_score: float
    explanation: str


class InteractionRequest(BaseModel):
    """Interaction event request."""
    user_id: str
    product_id: str
    interaction_type: str
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    engine_ready: bool
    timestamp: datetime


# ============================================
# FASTAPI APP SETUP
# ============================================

app = FastAPI(
    title="E-Commerce Recommendation Engine",
    description="Advanced AI-powered product recommendation system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global recommendation engine
recommendation_engine: Optional[RecommendationEngine] = None


# ============================================
# STARTUP/SHUTDOWN
# ============================================

@app.on_event("startup")
async def startup_event():
    """Initialize app on startup."""
    logger.info("Starting FastAPI application")
    
    global recommendation_engine
    
    try:
        # Initialize database
        db.initialize(use_sqlite=True, sqlite_path="data/ecommerce_rec.db")
        
        # Load or train recommendation engine
        logger.info("Loading recommendation engine")
        
        from src.data import DataLoader, DataValidator
        
        # Load processed data
        try:
            users = pd.read_csv("data/processed/users.csv")
            products = pd.read_csv("data/processed/products.csv")
            interactions = pd.read_csv("data/processed/interactions.csv")
        except FileNotFoundError:
            logger.warning("Processed data not found, loading raw data")
            users, products, interactions = DataLoader.load_all()
            users, products, interactions = DataValidator.validate_all(users, products, interactions)

        _synchronize_catalog(db.get_session(), users, products, interactions)
        
        # Initialize engine
        recommendation_engine = RecommendationEngine()
        recommendation_engine.initialize_models(users, products, interactions)
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


def _synchronize_catalog(db_session: Session, users, products, interactions) -> None:
    """Synchronize shipped CSV data into SQL for catalog and event endpoints."""
    try:
        for row in users.itertuples(index=False):
            user_id = str(row.user_id)
            if db_session.query(User).filter(User.user_id == user_id).first() is None:
                db_session.add(User(
                    user_id=user_id,
                    age=int(row.age),
                    gender=str(row.gender),
                    location=str(row.location),
                    signup_date=_parse_datetime(row.signup_date),
                ))
        for row in products.itertuples(index=False):
            product_id = str(row.product_id)
            if db_session.query(Product).filter(Product.product_id == product_id).first() is None:
                db_session.add(Product(
                    product_id=product_id,
                    product_name=str(row.product_name),
                    category=str(row.category),
                    subcategory=str(row.subcategory),
                    brand=str(row.brand),
                    description=str(row.description),
                    price=float(row.price),
                    rating=float(row.rating),
                    stock=int(row.stock),
                    created_at=_parse_datetime(row.created_at),
                ))
        db_session.commit()
        existing_interactions = db_session.query(Interaction.interaction_id).count()
        if existing_interactions == 0:
            records = [
                Interaction(
                    interaction_id=str(row.interaction_id),
                    user_id=str(row.user_id),
                    product_id=str(row.product_id),
                    interaction_type=str(row.interaction_type),
                    weight=_interaction_weight(str(row.interaction_type)),
                    timestamp=_parse_datetime(row.timestamp),
                )
                for row in interactions.itertuples(index=False)
            ]
            db_session.add_all(records)
            db_session.commit()
    except Exception:
        db_session.rollback()
        raise
    finally:
        db_session.close()


def _interaction_weight(interaction_type: str) -> float:
    return {
        "view": 1.0,
        "click": 2.0,
        "wishlist": 3.0,
        "cart": 4.0,
        "purchase": 5.0,
    }.get(interaction_type, 1.0)


def _parse_datetime(value) -> datetime:
    parsed = pd.to_datetime(value, errors="coerce")
    return parsed.to_pydatetime() if not pd.isna(parsed) else datetime.utcnow()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application")
    db.close()


# ============================================
# HEALTH & STATUS ENDPOINTS
# ============================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        engine_ready=recommendation_engine is not None and recommendation_engine.is_ready,
        timestamp=datetime.utcnow()
    )


@app.get("/status")
async def status():
    """Get system status and engine info."""
    if not recommendation_engine:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    return recommendation_engine.get_engine_status()


@app.post("/auth/register", response_model=AuthResponse, status_code=201)
async def register(request: RegisterRequest, db_session: Session = Depends(get_db)):
    """Register a customer or seller account."""
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", request.email):
        raise HTTPException(status_code=422, detail="A valid email address is required")
    if len(request.password) < 12 or not re.search(r"[A-Z]", request.password) or not re.search(r"\d", request.password):
        raise HTTPException(status_code=422, detail="Password must be 12+ characters with a number and uppercase letter")
    role = request.role.upper()
    if role not in {"CUSTOMER", "SELLER"}:
        raise HTTPException(status_code=422, detail="Public registration supports CUSTOMER or SELLER")
    if db_session.query(User).filter((User.email == request.email.lower()) | (User.user_id == request.user_id)).first():
        raise HTTPException(status_code=409, detail="Account already exists")
    user = User(
        user_id=request.user_id,
        email=request.email.lower(),
        password_hash=hash_password(request.password),
        role=role,
        age=request.age,
        gender=request.gender,
        location=request.location,
    )
    db_session.add(user)
    db_session.commit()
    return AuthResponse(access_token=create_access_token(user), user_id=user.user_id, role=role)


@app.post("/auth/login", response_model=AuthResponse)
async def login(request: LoginRequest, db_session: Session = Depends(get_db)):
    """Authenticate an account and issue a short-lived access token."""
    user = db_session.query(User).filter(User.email == request.email.lower()).first()
    if user is None or not user.password_hash or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is inactive")
    return AuthResponse(access_token=create_access_token(user), user_id=user.user_id, role=user.role)


@app.get("/auth/me", response_model=UserSchema)
async def current_profile(user: User = Depends(get_current_user)):
    """Return the authenticated profile."""
    return user


def _get_or_create_cart(user_id: str, db_session: Session) -> Cart:
    cart = db_session.query(Cart).filter(Cart.user_id == user_id).first()
    if cart is None:
        cart = Cart(user_id=user_id)
        db_session.add(cart)
        db_session.flush()
    return cart


def _cart_response(cart: Cart, db_session: Session) -> CartResponse:
    items = []
    total = 0.0
    for item in cart.items:
        product = db_session.query(Product).filter(Product.product_id == item.product_id).first()
        if product is None:
            continue
        subtotal = round(product.price * item.quantity, 2)
        total += subtotal
        items.append(CartItemResponse(
            product_id=product.product_id,
            product_name=product.product_name,
            quantity=item.quantity,
            unit_price=product.price,
            subtotal=subtotal,
        ))
    return CartResponse(items=items, total=round(total, 2))


@app.get("/cart", response_model=CartResponse)
async def get_cart(
    user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db),
):
    """Get the authenticated customer's persistent cart."""
    return _cart_response(_get_or_create_cart(user.user_id, db_session), db_session)


@app.post("/cart/items", response_model=CartResponse)
async def add_cart_item(
    request: CartItemRequest,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db),
):
    """Add an in-stock product to the authenticated customer's cart."""
    product = db_session.query(Product).filter(Product.product_id == request.product_id).first()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    cart = _get_or_create_cart(user.user_id, db_session)
    item = next((entry for entry in cart.items if entry.product_id == product.product_id), None)
    quantity = request.quantity if item is None else item.quantity + request.quantity
    if quantity > product.stock:
        raise HTTPException(status_code=409, detail="Requested quantity exceeds available stock")
    if item is None:
        cart.items.append(CartItem(product_id=product.product_id, quantity=quantity))
    else:
        item.quantity = quantity
    db_session.commit()
    return _cart_response(cart, db_session)


@app.delete("/cart/items/{product_id}", response_model=CartResponse)
async def remove_cart_item(
    product_id: str,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db),
):
    """Remove a product from the authenticated customer's cart."""
    cart = _get_or_create_cart(user.user_id, db_session)
    item = next((entry for entry in cart.items if entry.product_id == product_id), None)
    if item is None:
        raise HTTPException(status_code=404, detail="Cart item not found")
    db_session.delete(item)
    db_session.commit()
    return _cart_response(cart, db_session)


@app.get("/wishlist", response_model=List[ProductSchema])
async def get_wishlist(
    user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db),
):
    """List saved products for the authenticated customer."""
    ids = db_session.query(WishlistItem.product_id).filter(WishlistItem.user_id == user.user_id).all()
    product_ids = [row[0] for row in ids]
    if not product_ids:
        return []
    return db_session.query(Product).filter(Product.product_id.in_(product_ids)).all()


@app.post("/wishlist/{product_id}", response_model=ProductSchema, status_code=201)
async def add_to_wishlist(
    product_id: str,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db),
):
    """Save a product for the authenticated customer."""
    product = db_session.query(Product).filter(Product.product_id == product_id).first()
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    existing = db_session.query(WishlistItem).filter_by(user_id=user.user_id, product_id=product_id).first()
    if existing is None:
        db_session.add(WishlistItem(user_id=user.user_id, product_id=product_id))
        db_session.commit()
    return product


@app.delete("/wishlist/{product_id}", status_code=204)
async def remove_from_wishlist(
    product_id: str,
    user: User = Depends(get_current_user),
    db_session: Session = Depends(get_db),
):
    """Remove a saved product."""
    item = db_session.query(WishlistItem).filter_by(user_id=user.user_id, product_id=product_id).first()
    if item is None:
        raise HTTPException(status_code=404, detail="Wishlist item not found")
    db_session.delete(item)
    db_session.commit()


# ============================================
# PRODUCT ENDPOINTS
# ============================================

@app.get("/products", response_model=List[ProductSchema])
async def get_products(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    db_session: Session = Depends(get_db)
):
    """Get products list with pagination and filtering."""
    query = db_session.query(Product)
    
    if category:
        query = query.filter(Product.category == category)
    
    products = query.offset(skip).limit(limit).all()
    return products


@app.get("/products/{product_id}", response_model=ProductSchema)
async def get_product(product_id: str, db_session: Session = Depends(get_db)):
    """Get single product details."""
    product = db_session.query(Product).filter(Product.product_id == product_id).first()
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    return product


# ============================================
# USER ENDPOINTS
# ============================================

@app.get("/users/{user_id}", response_model=UserSchema)
async def get_user(user_id: str, db_session: Session = Depends(get_db)):
    """Get user profile."""
    user = db_session.query(User).filter(User.user_id == user_id).first()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user


@app.post("/users")
async def create_user(user: UserSchema, db_session: Session = Depends(get_db)):
    """Create new user."""
    # Check if user exists
    existing = db_session.query(User).filter(User.user_id == user.user_id).first()
    if existing:
        raise HTTPException(status_code=400, detail="User already exists")
    
    db_user = User(
        user_id=user.user_id,
        age=user.age,
        gender=user.gender,
        location=user.location
    )
    db_session.add(db_user)
    db_session.commit()
    
    return {"message": "User created", "user_id": user.user_id}


# ============================================
# RECOMMENDATION ENDPOINTS
# ============================================

@app.get("/recommendations/user/{user_id}", response_model=List[RecommendationResponse])
async def get_user_recommendations(
    user_id: str,
    n: int = Query(10, ge=1, le=50),
    category: Optional[str] = None,
    db_session: Session = Depends(get_db)
):
    """Get personalized recommendations for a user."""
    if not recommendation_engine or not recommendation_engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        recommendations = recommendation_engine.recommend_for_user(
            user_id=user_id,
            n_recommendations=n,
            category=category,
            exclude_purchased=True,
            min_rating=2.0
        )
        
        # Store recommendations in database
        for rec in recommendations:
            db_rec = Recommendation(
                user_id=user_id,
                product_id=rec['product_id'],
                recommendation_score=rec['recommendation_score'],
                explanation=rec['explanation'],
                category_filter=category,
                model_version='v1'
            )
            db_session.add(db_rec)
        
        db_session.commit()
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/product/{product_id}", response_model=List[ProductSchema])
async def get_similar_products(
    product_id: str,
    n: int = Query(5, ge=1, le=20),
    db_session: Session = Depends(get_db)
):
    """Get products similar to a given product."""
    if not recommendation_engine or not recommendation_engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        similar = recommendation_engine.similar_products(product_id, n)
        return similar
    except Exception as e:
        logger.error(f"Error getting similar products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/trending", response_model=List[ProductSchema])
async def get_trending_products(
    n: int = Query(10, ge=1, le=50),
    db_session: Session = Depends(get_db)
):
    """Get trending products."""
    if not recommendation_engine or not recommendation_engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        trending = recommendation_engine.trending_products(n)
        return trending
    except Exception as e:
        logger.error(f"Error getting trending products: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations/frequently-bought/{product_id}", response_model=List[ProductSchema])
async def get_frequently_bought(
    product_id: str,
    n: int = Query(5, ge=1, le=20),
    db_session: Session = Depends(get_db)
):
    """Get products frequently bought with a given product."""
    if not recommendation_engine or not recommendation_engine.is_ready:
        raise HTTPException(status_code=503, detail="Engine not ready")
    
    try:
        together = recommendation_engine.frequently_bought_together(product_id, n)
        return together
    except Exception as e:
        logger.error(f"Error getting frequently bought: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# EVENT TRACKING ENDPOINTS
# ============================================

@app.post("/events/interaction")
async def track_interaction(
    interaction: InteractionRequest,
    db_session: Session = Depends(get_db)
):
    """Track user-product interaction."""
    try:
        # Map interaction type to weight
        weight_map = {
            'view': 1,
            'click': 2,
            'wishlist': 3,
            'cart': 4,
            'purchase': 5
        }
        weight = weight_map.get(interaction.interaction_type, 1)
        
        # Create interaction record
        db_interaction = Interaction(
            interaction_id=f"{interaction.user_id}_{interaction.product_id}_{datetime.utcnow().timestamp()}",
            user_id=interaction.user_id,
            product_id=interaction.product_id,
            interaction_type=interaction.interaction_type,
            weight=weight,
            session_id=interaction.session_id
        )
        
        db_session.add(db_interaction)
        db_session.commit()
        
        return {"message": "Interaction tracked", "interaction_id": db_interaction.interaction_id}
    
    except Exception as e:
        logger.error(f"Error tracking interaction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/events/recommendation-click")
async def track_recommendation_click(
    user_id: str,
    product_id: str,
    db_session: Session = Depends(get_db)
):
    """Track recommendation click."""
    try:
        rec = db_session.query(Recommendation).filter(
            (Recommendation.user_id == user_id) &
            (Recommendation.product_id == product_id)
        ).first()
        
        if rec:
            rec.clicked = True
            db_session.commit()
        
        return {"message": "Click tracked"}
    except Exception as e:
        logger.error(f"Error tracking click: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# SEARCH ENDPOINT
# ============================================

@app.get("/search", response_model=List[ProductSchema])
async def search_products(
    q: str = Query(..., min_length=1),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db_session: Session = Depends(get_db)
):
    """Search products by name or description."""
    products = db_session.query(Product).filter(
        (Product.product_name.ilike(f"%{q}%")) |
        (Product.description.ilike(f"%{q}%"))
    ).offset(skip).limit(limit).all()
    
    return products


# ============================================
# CATEGORIES ENDPOINT
# ============================================

@app.get("/categories")
async def get_categories(db_session: Session = Depends(get_db)):
    """Get all product categories."""
    categories = db_session.query(Product.category).distinct().all()
    return [cat[0] for cat in categories if cat[0]]


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level="info"
    )
