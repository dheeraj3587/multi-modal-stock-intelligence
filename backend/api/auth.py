"""
Authentication API - signup, login (JWT), and Upstox OAuth helpers.
Passwords are hashed with bcrypt; JWTs are used for session management.
"""

from datetime import datetime, timedelta
from typing import Optional

import os
import requests
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from backend.db.session import get_db
from backend.db.models import User
from backend.utils.jwt_utils import create_access_token, verify_token

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

UPSTOX_API_KEY = os.getenv("UPSTOX_API_KEY", "")
UPSTOX_API_SECRET = os.getenv("UPSTOX_API_SECRET", "")
UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8000/auth/upstox/callback")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173")


# --- Request / response models ---------------------------------------

class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str
    full_name: Optional[str] = None


class UserProfile(BaseModel):
    email: str
    full_name: Optional[str] = None
    is_active: bool
    created_at: str
    upstox_connected: bool


# --- Helper - get current user from JWT ------------------------------

def get_current_user(
    authorization: Optional[str] = Depends(lambda authorization=None: authorization),
    db: Session = Depends(get_db),
) -> User:
    """Extract and validate JWT from the Authorization header."""
    # FastAPI Header dependency
    from fastapi import Header
    raise HTTPException(status_code=401, detail="Not implemented inline")


from fastapi import Header

def _current_user(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> User:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    token = authorization[7:]
    payload = verify_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    user_id: str = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


# --- Signup -----------------------------------------------------------

def _safe_hash(password: str) -> str:
    """Hash password, truncating to 72 bytes for bcrypt compat."""
    return pwd_context.hash(password[:72])


def _safe_verify(password: str, hashed: str) -> bool:
    """Verify password, truncating to 72 bytes for bcrypt compat."""
    return pwd_context.verify(password[:72], hashed)


@router.post("/signup", response_model=AuthResponse)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    """Register a new user account."""
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=409, detail="Email already registered")

    hashed = _safe_hash(body.password)
    user = User(email=body.email, hashed_password=hashed, full_name=body.full_name)
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.id, "email": user.email})
    return AuthResponse(access_token=token, email=user.email, full_name=user.full_name)


# --- Login ------------------------------------------------------------

@router.post("/login", response_model=AuthResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    """Authenticate with email + password and receive a JWT."""
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not _safe_verify(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    token = create_access_token({"sub": user.id, "email": user.email})
    return AuthResponse(access_token=token, email=user.email, full_name=user.full_name)


# --- Profile ----------------------------------------------------------

@router.get("/profile", response_model=UserProfile)
def get_profile(user: User = Depends(_current_user)):
    """Return the authenticated user's profile."""
    return UserProfile(
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        created_at=user.created_at.isoformat() if user.created_at else "",
        upstox_connected=bool(user.upstox_access_token),
    )


# --- Upstox OAuth (kept as-is, now also persists token to DB) --------

@router.get("/upstox/login")
def upstox_login():
    """Redirect user to Upstox OAuth consent page."""
    if not UPSTOX_API_KEY:
        raise HTTPException(status_code=500, detail="UPSTOX_API_KEY is not set")
    url = (
        f"https://api.upstox.com/v2/login/authorization/dialog"
        f"?response_type=code&client_id={UPSTOX_API_KEY}&redirect_uri={UPSTOX_REDIRECT_URI}"
    )
    return RedirectResponse(url)


@router.get("/upstox/callback")
def upstox_callback(code: str, request: Request, db: Session = Depends(get_db)):
    """
    Exchange Upstox authorization code for an access token.
    If called from a browser navigation, redirect to the frontend with the token.
    """
    sec_fetch_mode = request.headers.get("sec-fetch-mode", "")
    sec_fetch_dest = request.headers.get("sec-fetch-dest", "")
    if sec_fetch_mode == "navigate" or sec_fetch_dest == "document":
        return RedirectResponse(f"{FRONTEND_BASE_URL}/auth/callback?code={code}")

    if not UPSTOX_API_KEY or not UPSTOX_API_SECRET:
        raise HTTPException(status_code=500, detail="Upstox credentials not configured")

    token_url = "https://api.upstox.com/v2/login/authorization/token"
    headers = {"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "code": code,
        "client_id": UPSTOX_API_KEY,
        "client_secret": UPSTOX_API_SECRET,
        "redirect_uri": UPSTOX_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    resp = requests.post(token_url, headers=headers, data=data)
    if resp.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Upstox token exchange failed: {resp.text}")

    json_resp = resp.json()
    access_token = json_resp.get("access_token")

    # Persist the server-side Upstox token for background data fetching.
    # If the request includes an app JWT we can attribute the token to the user.
    app_jwt = request.headers.get("X-App-Token") or request.query_params.get("app_token")
    if app_jwt and access_token:
        payload = verify_token(app_jwt)
        if payload:
            user = db.query(User).filter(User.id == payload.get("sub")).first()
            if user:
                user.upstox_access_token = access_token
                # Upstox tokens expire at 03:30 AM next day
                user.upstox_token_expiry = (
                    datetime.utcnow().replace(hour=22, minute=0, second=0)  # ~03:30 IST
                )
                db.commit()
                logger.info(f"Stored Upstox token for user {user.email}")

    # Also store a global "server" token in Redis so the scheduler can use it
    _store_global_upstox_token(access_token)

    return json_resp


def _store_global_upstox_token(token: str):
    """Persist the latest Upstox access token in Redis for background jobs."""
    if not token:
        return
    try:
        import redis as _redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = _redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        # TTL ~ 24 h (Upstox tokens last until 03:30 AM next day)
        r.setex("upstox_global_token", 86400, token)
        logger.info("Stored global Upstox access token in Redis")
    except Exception as e:
        logger.warning(f"Could not store global Upstox token in Redis: {e}")


def get_global_upstox_token() -> Optional[str]:
    """Retrieve the latest global Upstox access token from Redis."""
    try:
        import redis as _redis
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        r = _redis.from_url(url, decode_responses=True, socket_connect_timeout=2)
        return r.get("upstox_global_token")
    except Exception:
        return None
