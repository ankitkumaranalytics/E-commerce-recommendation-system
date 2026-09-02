"""
Synthetic E-commerce Dataset Generator

Generates realistic datasets for development and testing:
- users.csv
- products.csv
- interactions.csv
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

def generate_users(n_users: int = 1000, output_path: str = "data/raw/users.csv") -> None:
    """Generate synthetic user data."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    locations = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", 
                 "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
                 "London", "Delhi", "Tokyo", "Shanghai", "Mumbai"]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'age', 'gender', 'location', 'signup_date'])
        
        for i in range(1, n_users + 1):
            age = random.randint(18, 75)
            gender = random.choice(['M', 'F', 'Other'])
            location = random.choice(locations)
            signup_date = (datetime.now() - timedelta(days=random.randint(1, 365))).strftime('%Y-%m-%d')
            writer.writerow([f'U{i:05d}', age, gender, location, signup_date])
    
    print(f"[OK] Generated {n_users} users -> {output_path}")

def generate_products(n_products: int = 500, output_path: str = "data/raw/products.csv") -> None:
    """Generate synthetic product data."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    categories = ["Electronics", "Fashion", "Home & Garden", "Sports", "Books", "Toys", "Food", "Beauty"]
    
    subcategories = {
        "Electronics": ["Smartphones", "Laptops", "Tablets", "Accessories"],
        "Fashion": ["Men's Clothing", "Women's Clothing", "Shoes", "Accessories"],
        "Home & Garden": ["Furniture", "Bedding", "Kitchen", "Decor"],
        "Sports": ["Athletic Wear", "Equipment", "Footwear", "Accessories"],
        "Books": ["Fiction", "Non-Fiction", "Self-Help", "Educational"],
        "Toys": ["Action Figures", "Board Games", "Puzzles", "Building Blocks"],
        "Food": ["Snacks", "Beverages", "Organic", "Specialty"],
        "Beauty": ["Skincare", "Haircare", "Makeup", "Fragrances"]
    }
    
    brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", "Premium", "Luxury", "Budget", "Elite", "Standard"]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['product_id', 'product_name', 'category', 'subcategory', 'brand', 
                        'description', 'price', 'rating', 'stock', 'created_at'])
        
        for i in range(1, n_products + 1):
            product_id = f'P{i:05d}'
            category = random.choice(categories)
            subcategory = random.choice(subcategories[category])
            brand = random.choice(brands)
            product_name = f"{brand} {category} {i}"
            description = f"High-quality {category.lower()} product. Features durability, style, and value."
            price = round(random.uniform(10, 1000), 2)
            rating = round(random.uniform(2.0, 5.0), 1)
            stock = random.randint(0, 1000)
            created_at = (datetime.now() - timedelta(days=random.randint(1, 180))).strftime('%Y-%m-%d')
            
            writer.writerow([product_id, product_name, category, subcategory, brand, 
                           description, price, rating, stock, created_at])
    
    print(f"[OK] Generated {n_products} products -> {output_path}")

def generate_interactions(n_users: int = 1000, n_products: int = 500, 
                         n_interactions: int = 10000, output_path: str = "data/raw/interactions.csv") -> None:
    """Generate synthetic user-product interaction data with realistic distribution."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    interaction_types = ["view", "click", "wishlist", "cart", "purchase"]
    
    # Realistic distribution: more views than clicks, fewer purchases
    interaction_distribution = {
        "view": 0.50,
        "click": 0.25,
        "wishlist": 0.10,
        "cart": 0.10,
        "purchase": 0.05
    }
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['interaction_id', 'user_id', 'product_id', 'interaction_type', 'timestamp'])
        
        for i in range(1, n_interactions + 1):
            interaction_id = f'I{i:07d}'
            user_id = f'U{random.randint(1, n_users):05d}'
            product_id = f'P{random.randint(1, n_products):05d}'
            
            # Realistic distribution
            rand = random.random()
            cumsum = 0
            interaction_type = "view"
            for itype, prob in interaction_distribution.items():
                cumsum += prob
                if rand <= cumsum:
                    interaction_type = itype
                    break
            
            # Timestamp within last 90 days
            timestamp = (datetime.now() - timedelta(days=random.randint(0, 90))).strftime('%Y-%m-%d %H:%M:%S')
            
            writer.writerow([interaction_id, user_id, product_id, interaction_type, timestamp])
    
    print(f"[OK] Generated {n_interactions} interactions -> {output_path}")

def generate_all_datasets(n_users: int = 1000, n_products: int = 500, n_interactions: int = 10000) -> None:
    """Generate all datasets."""
    print("\n" + "="*60)
    print("Generating Synthetic E-commerce Dataset")
    print("="*60 + "\n")
    
    generate_users(n_users)
    generate_products(n_products)
    generate_interactions(n_users, n_products, n_interactions)
    
    print("\n" + "="*60)
    print("[OK] Dataset generation complete!")
    print("="*60 + "\n")

if __name__ == "__main__":
    # Default values
    n_users = 1000
    n_products = 500
    n_interactions = 10000
    
    # Parse command line arguments if provided
    if len(sys.argv) > 1:
        n_users = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_products = int(sys.argv[2])
    if len(sys.argv) > 3:
        n_interactions = int(sys.argv[3])
    
    generate_all_datasets(n_users, n_products, n_interactions)
