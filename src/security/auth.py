# src/security/auth.py
import os
import hashlib
import requests
from langgraph_sdk.auth import is_studio_user, Auth

auth = Auth()

# --- CONFIG ---
# Pode sobrescrever via env se precisar apontar para outro endpoint (ex.: staging)
LANGSMITH_API = os.getenv("LANGSMITH_API", "https://api.smith.langchain.com")
# Endpoint leve para validar a key. Basta retornar 200 para considerar válida.
VALIDATION_URL = f"{LANGSMITH_API}/api/v1/sessions"

def is_valid_key(api_key: str) -> bool:
    # Fallback local para testes (ex.: cURL)
    return api_key == "123"

def _get_header(headers: dict, name: str):
    """Busca header em str ou bytes e decodifica com segurança."""
    val = headers.get(name) or headers.get(name.encode())
    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return val.decode("utf-8", errors="ignore")
    return val

def _identity_from_key_prefix(key: str) -> str:
    # Identidade estável mas não reveladora: prefixo do hash da key
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:10]
    return f"studio-{h}"

def _validate_langsmith_key(x_api_key: str) -> str | None:
    """
    Valida a key do LangSmith chamando a API oficial com o mesmo header.
    Retorna uma identity (string) se for válida; None caso contrário.
    """
    try:
        r = requests.get(VALIDATION_URL, headers={"X-Api-Key": x_api_key}, timeout=5)
        if r.status_code == 200:
            # Opcional: você pode consultar outro endpoint para pegar e-mail/ID real.
            # Aqui usamos um ID derivado da própria chave, sem expor a chave.
            return _identity_from_key_prefix(x_api_key)
        return None
    except Exception:
        return None

@auth.authenticate
async def authenticate(headers: dict) -> Auth.types.MinimalUserDict:
    # 1) Caminho Studio: aceita X-Api-Key e valida contra LangSmith
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

    # 2) Fallback local: sua API key simples para scripts/cURL
    api_key = _get_header(headers, "x-api-key") or _get_header(headers, "authorization")
    # Suporta "Authorization: Bearer 123" ou "X-Api-Key: 123"
    if api_key:
        # Normaliza caso venha como "Bearer 123"
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
    Owner-only por padrão. Se o usuário for detectado como Studio (via SDK),
    não aplica filtros (útil para depurar pelo Studio).
    """
    if is_studio_user(ctx.user):
        return {}

    filters = {"owner": ctx.user.identity}
    metadata = value.setdefault("metadata", {})
    metadata.update(filters)
    return filters
