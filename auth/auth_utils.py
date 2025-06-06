import json
import hashlib
import os
import jwt
import datetime
from typing import Optional, Tuple

USER_DB_PATH = "auth/user_db.json"
JWT_SECRET = "your_super_secret_key"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600  # 1 hour

def initialize_user_db():
    """Create user database if it doesn't exist"""
    if not os.path.exists(USER_DB_PATH):
        os.makedirs(os.path.dirname(USER_DB_PATH), exist_ok=True)
        with open(USER_DB_PATH, "w") as f:
            json.dump({"users": {}}, f)

def hash_password(password: str) -> str:
    """SHA-256 password hashing with salt"""
    salt = "diabetes_app_salt"
    return hashlib.sha256((password + salt).encode()).hexdigest()

def register_user(username: str, password: str) -> bool:
    """Register new user"""
    initialize_user_db()
    
    with open(USER_DB_PATH, "r+") as f:
        db = json.load(f)
        if username in db["users"]:
            return False
        
        db["users"][username] = {
            "password_hash": hash_password(password),
            "predictions": []
        }
        
        f.seek(0)
        json.dump(db, f, indent=4)
        f.truncate()
        return True

def verify_user(username: str, password: str) -> bool:
    """Verify login credentials"""
    if not os.path.exists(USER_DB_PATH):
        return False
    
    with open(USER_DB_PATH, "r") as f:
        db = json.load(f)
        user = db["users"].get(username)
        if not user:
            return False
        
        return user["password_hash"] == hash_password(password)

def update_password(username: str, old_password: str, new_password: str) -> Tuple[bool, str]:
    """Update a user's password. Returns (status, message)."""
    if not os.path.exists(USER_DB_PATH):
        return False, "User database does not exist."

    try:
        with open(USER_DB_PATH, "r+") as f:
            db = json.load(f)
            user = db["users"].get(username)

            if not user:
                return False, "User not found."

            if user["password_hash"] != hash_password(old_password):
                return False, "Old password is incorrect."

            db["users"][username]["password_hash"] = hash_password(new_password)

            f.seek(0)
            json.dump(db, f, indent=4)
            f.truncate()
            return True, "Password updated successfully."
    except (json.JSONDecodeError, IOError) as e:
        return False, f"Error updating password: {e}"

def delete_user(username: str, password: str) -> Tuple[bool, str]:
    """Delete a user from the database. Returns (status, message)."""
    if not os.path.exists(USER_DB_PATH):
        return False, "User database does not exist."

    try:
        with open(USER_DB_PATH, "r+") as f:
            db = json.load(f)
            user = db["users"].get(username)

            if not user:
                return False, "User not found."

            if user["password_hash"] != hash_password(password):
                return False, "Password incorrect."

            del db["users"][username]

            f.seek(0)
            json.dump(db, f, indent=4)
            f.truncate()
            return True, "User deleted successfully."
    except (json.JSONDecodeError, IOError) as e:
        return False, f"Error deleting user: {e}"

def generate_token(username: str) -> str:
    """Generate JWT token for authenticated user"""
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def verify_token(token: str) -> Tuple[bool, Optional[str]]:
    """Verify JWT token"""
    try:
        decoded = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return True, decoded["username"]
    except jwt.ExpiredSignatureError:
        return False, "Token has expired"
    except jwt.InvalidTokenError:
        return False, "Invalid token"
