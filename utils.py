import secrets

def generate_api_key() -> str:
    return f"omni_{secrets.token_urlsafe(32)}"
