from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
import os
import requests
from pydantic import BaseModel

router = APIRouter()

UPSTOX_API_KEY = os.getenv("UPSTOX_API_KEY")
UPSTOX_API_SECRET = os.getenv("UPSTOX_API_SECRET")
# By default, use the backend callback (typical OAuth pattern).
# If you want Upstox to redirect directly to the frontend, set UPSTOX_REDIRECT_URI explicitly.
UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "http://localhost:8000/auth/upstox/callback")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173")

class TokenResponse(BaseModel):
    access_token: str
    extended_token: str | None = None

@router.get("/auth/upstox/login")
def login():
    """
    Redirects the user to Upstox login page.
    """
    if not UPSTOX_API_KEY:
        raise HTTPException(status_code=500, detail="UPSTOX_API_KEY is not set")
    
    # Official Upstox OAuth URL format
    # https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={api_key}&redirect_uri={redirect_uri}
    url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={UPSTOX_API_KEY}&redirect_uri={UPSTOX_REDIRECT_URI}"
    return RedirectResponse(url)

@router.get("/auth/upstox/callback")
def callback(code: str, request: Request):
    """
    Exchanges authorization code for an access token.
    For this simplified flow, we return the token directly to the frontend (or let frontend call this via proxy).
    Typically, frontend calls this endpoint with the code.
    """
    # If Upstox redirects the browser to this backend endpoint (common when redirect_uri is set to backend),
    # bounce to the frontend callback route so the SPA can complete the flow and store the token.
    # We detect a top-level navigation via Sec-Fetch headers.
    sec_fetch_mode = request.headers.get("sec-fetch-mode", "")
    sec_fetch_dest = request.headers.get("sec-fetch-dest", "")
    if sec_fetch_mode == "navigate" or sec_fetch_dest == "document":
        return RedirectResponse(f"{FRONTEND_BASE_URL}/auth/callback?code={code}")

    if not UPSTOX_API_KEY or not UPSTOX_API_SECRET:
         raise HTTPException(status_code=500, detail="Upstox credentials not configured")

    url = "https://api.upstox.com/v2/login/authorization/token"
    headers = {
        "accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "code": code,
        "client_id": UPSTOX_API_KEY,
        "client_secret": UPSTOX_API_SECRET,
        "redirect_uri": UPSTOX_REDIRECT_URI,
        "grant_type": "authorization_code"
    }

    response = requests.post(url, headers=headers, data=data)
    
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to get token: {response.text}")

    json_response = response.json()
    return json_response
