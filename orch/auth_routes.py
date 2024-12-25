from fastapi import APIRouter, HTTPException, Depends, status, BackgroundTasks, Request
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import datetime, timedelta
import uuid
from pymongo import MongoClient
from typing import Optional
import pyotp
from fastapi_limiter.depends import RateLimiter
from fastapi_limiter import FastAPILimiter
import redis.asyncio as aioredis
import logging

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Rate limiter initialization
redis_url = "redis://localhost:6379"
app.state.redis = await aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
await FastAPILimiter.init(app.state.redis)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler("auth_events.log"), logging.StreamHandler()])
logger = logging.getLogger("AuthEvents")

class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)

class PasswordResetRequest(BaseModel):
    email: EmailStr

class PasswordResetComplete(BaseModel):
    token: str
    new_password: str = Field(min_length=8)

class UserProfileUpdate(BaseModel):
    full_name: Optional[str]
    dietary_preferences: Optional[str]

class ChangePassword(BaseModel):
    old_password: str
    new_password: str = Field(min_length=8)

class MFAEnableRequest(BaseModel):
    secret: str

class MFAVerifyRequest(BaseModel):
    code: str

def validate_password_complexity(password: str) -> bool:
    if len(password) < 8 or not any(char.isupper() for char in password) or not any(char.isdigit() for char in password):
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters, include an uppercase letter, and a number.")
    return True

@router.post("/auth/register", dependencies=[Depends(RateLimiter(times=5, seconds=60))])
async def register_user(user: UserRegister, db: MongoClient = Depends(get_db)):
    if db.users.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="User already registered")
    validate_password_complexity(user.password)
    hashed_password = pwd_context.hash(user.password)
    db.users.insert_one({"id": str(uuid.uuid4()), "email": user.email, "hashed_password": hashed_password, "is_active": True, "is_verified": False, "created_at": datetime.utcnow(), "failed_attempts": 0, "devices": []})
    logger.info(f"User {user.email} registered.")
    return {"message": "User registered successfully"}

@router.post("/auth/login", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def login_user(email: EmailStr, password: str, request: Request, db: MongoClient = Depends(get_db)):
    user = db.users.find_one({"email": email})
    if not user or not pwd_context.verify(password, user["hashed_password"]):
        if user:
            db.users.update_one({"email": email}, {"$inc": {"failed_attempts": 1}})
            if user["failed_attempts"] >= 5:
                raise HTTPException(status_code=403, detail="Too many failed login attempts. IP blocked.")
        logger.warning(f"Failed login attempt for {email} from {request.client.host}")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if user["mfa_enabled"]:
        raise HTTPException(status_code=200, detail="MFA required")

    db.users.update_one({"email": email}, {"$set": {"failed_attempts": 0}, "$push": {"devices": {"ip": request.client.host, "last_login": datetime.utcnow()}}})
    access_token = create_access_token(data={"sub": email})
    logger.info(f"User {email} logged in successfully from {request.client.host}")
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/auth/enable-mfa", dependencies=[Depends(get_current_user)])
async def enable_mfa(request: MFAEnableRequest, current_user: dict = Depends(get_current_user), db: MongoClient = Depends(get_db)):
    totp = pyotp.TOTP(request.secret)
    db.users.update_one({"email": current_user["email"]}, {"$set": {"mfa_secret": request.secret, "mfa_enabled": True}})
    logger.info(f"MFA enabled for {current_user['email']}")
    return {"message": "MFA enabled"}

@router.post("/auth/verify-mfa", dependencies=[Depends(get_current_user)])
async def verify_mfa(request: MFAVerifyRequest, current_user: dict = Depends(get_current_user), db: MongoClient = Depends(get_db)):
    user = db.users.find_one({"email": current_user["email"]})
    if not user or not user.get("mfa_secret"):
        raise HTTPException(status_code=400, detail="MFA not enabled for this user")
    totp = pyotp.TOTP(user["mfa_secret"])
    if not totp.verify(request.code):
        logger.warning(f"Invalid MFA code for {current_user['email']}")
        raise HTTPException(status_code=401, detail="Invalid MFA code")
    logger.info(f"MFA verified for {current_user['email']}")
    return {"message": "MFA verified"}

@router.delete("/auth/me", dependencies=[Depends(get_current_user)])
async def delete_user_account(current_user: dict = Depends(get_current_user), db: MongoClient = Depends(get_db)):
    db.users.delete_one({"email": current_user["email"]})
    logger.info(f"User {current_user['email']} deleted their account")
    return {"message": "User account deleted"}
