# src/security/auth.py
import os
import hashlib
import requests
from langgraph_sdk.auth import is_studio_user, Auth

auth = Auth()

# --- CONFIG ---
# Can be overridden via environment variables if you need to point to another endpoint (e.g., staging)
LANGSMITH_API = os.getenv("LANGSMITH_API", "https://api.smith.langchain.com")
# Lightweight endpoint to validate the key. A 200 response means the key is considered valid.
VALIDATION_URL = f"{LANGSMITH_API}/api/v1/sessions"

def is_valid_key(api_key: str) -> bool:
    # Local fallback for tests (e.g., cURL)
    return api_key == "123"

def _get_header(headers: dict, name: str):
    """Get a header from str or bytes and decode safely."""
    val = headers.get(name) or headers.get(name.encode())
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return val.decode("utf-8", errors="ignore")
    return val

def _identity_from_key_prefix(key: str) -> str:
    # Stable but non-revealing identity: prefix of the key hash
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"studio-{h}"

def _validate_langsmith_key(x_api_key: str) -> str | None:
    """
    Validate the LangSmith key by calling the official API with the same header.
    Returns an identity string if valid; None otherwise.
    """
    try:
        r = requests.get(VALIDATION_URL, headers={"X-Api-Key": x_api_key}, timeout=5)
        if r.status_code == 200:
            # Optional: you could call another endpoint to retrieve email/real ID.
            # Here we use an ID derived from the key itself, without exposing the key.
            return _identity_from_key_prefix(x_api_key)
        return None
    except Exception:
        return None

@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    # 1) Studio path: accept X-Api-Key and validate against LangSmith
    x_api_key = _get_header(headers, "x-api-key")
    if x_api_key:
        identity = _validate_langsmith_key(x_api_key)
        if identity:
            return {
                "identity": identity,
                "is_authenticated": True,
            }
        # Se a key veio mas é inválida, rejeita explicitamente
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid LangSmith X-Api-Key")

    # 2) Local fallback: your simple API key for scripts/cURL
    api_key = _get_header(headers, "x-api-key") or _get_header(headers, "authorization")
    # Supports "Authorization: Bearer 123" or "X-Api-Key: 123"
    if api_key:
        # Normaliza caso venha como "Bearer 123"
        # Normalize if provided as "Bearer 123"
        if api_key.lower().startswith("bearer "):
            api_key = api_key.split(" ", 1)[1].strip()

    if not api_key or not is_valid_key(api_key):
        raise Auth.exceptions.HTTPException(status_code=401, detail="Invalid API key")

    return {
        "identity": "marim",
        "is_authenticated": True,
    }

@auth.on
async def add_owner(
    ctx: Auth.types.AuthContext,
    value: dict
) -> dict:
    """
    Owner-only by default. If the user is detected as Studio (via SDK),
    do not apply filters (useful for debugging via Studio).
    """
    if is_studio_user(ctx.user):
        return {}

    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
