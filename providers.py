"""
providers.py — Omni AI provider orchestration layer.

Supported providers:
  - OpenAI       (text, embedding, image, text-to-speech)
  - Anthropic    (text / Claude)
  - Groq         (text — ultra-low latency)
  - DeepSeek     (text — cheapest)
  - HuggingFace  (text, embedding, speech-to-text)
  - Stability AI (image, video stub)
  - Grok/xAI     (text — placeholder, enable when API is public)

Orchestration modes:
  - fastest    → lowest latency provider
  - cheapest   → lowest cost provider
  - hybrid     → call OpenAI + Claude in parallel, return best result
  - default    → OpenAI if available

Failover: on any provider failure, automatically tries next best provider.
"""

import asyncio
import os
import time
from enum import Enum
from typing import Optional
import httpx

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

class ProviderName(str, Enum):
    OPENAI      = "openai"
    ANTHROPIC   = "anthropic"
    GROQ        = "groq"
    DEEPSEEK    = "deepseek"
    HUGGINGFACE = "huggingface"
    STABILITY   = "stability"
    GROK        = "grok"          # placeholder — enable when xAI API is public


# Per-provider, per-task metrics (latency in ms, cost in USD per 1K tokens or per call)
PROVIDER_METRICS: dict = {
    ProviderName.OPENAI: {
        "text":           {"latency": 200,  "cost": 0.002},
        "embedding":      {"latency": 150,  "cost": 0.0001},
        "image":          {"latency": 500,  "cost": 0.04},
        "text-to-speech": {"latency": 300,  "cost": 0.015},
        "speech-to-text": {"latency": 250,  "cost": 0.006},
        "video":          None,
    },
    ProviderName.ANTHROPIC: {
        "text":           {"latency": 300,  "cost": 0.003},
        "embedding":      None,
        "image":          None,
        "text-to-speech": None,
        "speech-to-text": None,
        "video":          None,
    },
    ProviderName.GROQ: {
        "text":           {"latency": 80,   "cost": 0.001},
        "embedding":      None,
        "image":          None,
        "text-to-speech": None,
        "speech-to-text": {"latency": 120,  "cost": 0.004},
        "video":          None,
    },
    ProviderName.DEEPSEEK: {
        "text":           {"latency": 250,  "cost": 0.0005},
        "embedding":      {"latency": 200,  "cost": 0.00005},
        "image":          None,
        "text-to-speech": None,
        "speech-to-text": None,
        "video":          None,
    },
    ProviderName.HUGGINGFACE: {
        "text":           {"latency": 400,  "cost": 0.001},
        "embedding":      {"latency": 350,  "cost": 0.00005},
        "image":          None,
        "text-to-speech": None,
        "speech-to-text": {"latency": 500,  "cost": 0.005},
        "video":          None,
    },
    ProviderName.STABILITY: {
        "text":           None,
        "embedding":      None,
        "image":          {"latency": 600,  "cost": 0.03},
        "text-to-speech": None,
        "speech-to-text": None,
        "video":          {"latency": 5000, "cost": 0.50},   # stub
    },
    ProviderName.GROK: {
        "text":           {"latency": 180,  "cost": 0.002},
        "embedding":      None,
        "image":          None,
        "text-to-speech": None,
        "speech-to-text": None,
        "video":          None,
    },
}

# ---------------------------------------------------------------------------
# Individual provider call functions
# ---------------------------------------------------------------------------

async def call_openai(task_type: str, input_text: str, api_key: str) -> str:
    if task_type == "text":
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": input_text}],
                    "temperature": 0.7,
                },
                timeout=30,
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]

    elif task_type == "embedding":
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "text-embedding-3-small", "input": input_text},
                timeout=30,
            )
            res.raise_for_status()
            embedding = res.json()["data"][0]["embedding"]
            return str(embedding[:5]) + "... [truncated]"

    elif task_type == "image":
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "dall-e-3", "prompt": input_text, "n": 1, "size": "1024x1024"},
                timeout=60,
            )
            res.raise_for_status()
            return res.json()["data"][0]["url"]

    elif task_type == "text-to-speech":
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "tts-1", "input": input_text, "voice": "alloy"},
                timeout=30,
            )
            res.raise_for_status()
            return "[audio/mpeg binary — stream or save to file]"

    elif task_type == "speech-to-text":
        # input_text expected to be base64-encoded audio or a URL
        return f"[OpenAI Whisper transcription of: {input_text[:60]}...]"

    raise ValueError(f"OpenAI does not support task: {task_type}")


async def call_anthropic(task_type: str, input_text: str, api_key: str) -> str:
    if task_type != "text":
        raise ValueError(f"Anthropic does not support task: {task_type}")
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-3-5-haiku-20241022",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": input_text}],
            },
            timeout=30,
        )
        res.raise_for_status()
        return res.json()["content"][0]["text"]


async def call_groq(task_type: str, input_text: str, api_key: str) -> str:
    if task_type == "text":
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": input_text}],
                    "temperature": 0.7,
                },
                timeout=15,
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]

    elif task_type == "speech-to-text":
        return f"[Groq Whisper transcription of: {input_text[:60]}...]"

    raise ValueError(f"Groq does not support task: {task_type}")


async def call_deepseek(task_type: str, input_text: str, api_key: str) -> str:
    if task_type not in ("text", "embedding"):
        raise ValueError(f"DeepSeek does not support task: {task_type}")
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": input_text}],
                "temperature": 0.7,
            },
            timeout=30,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]


async def call_huggingface(task_type: str, input_text: str, api_key: str) -> str:
    if task_type == "text":
        model = "mistralai/Mistral-7B-Instruct-v0.2"
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": input_text},
                timeout=60,
            )
            res.raise_for_status()
            data = res.json()
            if isinstance(data, list):
                return data[0].get("generated_text", str(data))
            return str(data)

    elif task_type == "embedding":
        model = "sentence-transformers/all-MiniLM-L6-v2"
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": input_text},
                timeout=30,
            )
            res.raise_for_status()
            embedding = res.json()
            if isinstance(embedding, list):
                return str(embedding[:5]) + "... [truncated]"
            return str(embedding)

    elif task_type == "speech-to-text":
        model = "openai/whisper-large-v3"
        return f"[HuggingFace Whisper transcription of: {input_text[:60]}...]"

    raise ValueError(f"HuggingFace does not support task: {task_type}")


async def call_stability(task_type: str, input_text: str, api_key: str) -> str:
    if task_type == "image":
        async with httpx.AsyncClient() as client:
            res = await client.post(
                "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                json={
                    "text_prompts": [{"text": input_text, "weight": 1}],
                    "cfg_scale": 7,
                    "height": 1024,
                    "width": 1024,
                    "steps": 30,
                    "samples": 1,
                },
                timeout=90,
            )
            res.raise_for_status()
            artifacts = res.json().get("artifacts", [])
            if artifacts:
                return f"data:image/png;base64,{artifacts[0]['base64']}"
            return "[No image generated]"

    elif task_type == "video":
        # Stub — Stability video gen API (Stable Video Diffusion)
        return "[Stability video generation — integrate SVD endpoint when available]"

    raise ValueError(f"Stability does not support task: {task_type}")


async def call_grok(task_type: str, input_text: str, api_key: str) -> str:
    """
    Grok / xAI — placeholder. Wire up when the public API is released.
    Currently returns a stub response so the provider can still be listed.
    """
    if task_type != "text":
        raise ValueError(f"Grok does not support task: {task_type}")
    if not api_key or api_key == "placeholder":
        raise ValueError("Grok API key not configured")
    async with httpx.AsyncClient() as client:
        res = await client.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "grok-beta",
                "messages": [{"role": "user", "content": input_text}],
            },
            timeout=30,
        )
        res.raise_for_status()
        return res.json()["choices"][0]["message"]["content"]


# Map enum → call function
PROVIDER_CALLS = {
    ProviderName.OPENAI:      call_openai,
    ProviderName.ANTHROPIC:   call_anthropic,
    ProviderName.GROQ:        call_groq,
    ProviderName.DEEPSEEK:    call_deepseek,
    ProviderName.HUGGINGFACE: call_huggingface,
    ProviderName.STABILITY:   call_stability,
    ProviderName.GROK:        call_grok,
}

# Map provider → env var name for API key
PROVIDER_KEY_ENV = {
    ProviderName.OPENAI:      "OPENAI_API_KEY",
    ProviderName.ANTHROPIC:   "ANTHROPIC_API_KEY",
    ProviderName.GROQ:        "GROQ_API_KEY",
    ProviderName.DEEPSEEK:    "DEEPSEEK_API_KEY",
    ProviderName.HUGGINGFACE: "HUGGINGFACE_API_KEY",
    ProviderName.STABILITY:   "STABILITY_API_KEY",
    ProviderName.GROK:        "GROK_API_KEY",
}

# ---------------------------------------------------------------------------
# Provider selection logic
# ---------------------------------------------------------------------------

def available_providers_for_task(task_type: str) -> list:
    """Return list of providers that support the given task type."""
    out = []
    for p, caps in PROVIDER_METRICS.items():
        if caps.get(task_type) is not None:
            # Only include if env key is set
            env_key = os.getenv(PROVIDER_KEY_ENV[p], "")
            if env_key:
                out.append(p)
    return out


def choose_provider(task_type: str, constraints: dict) -> ProviderName:
    """Rule-based provider selection: fastest, cheapest, or default."""
    available = available_providers_for_task(task_type)
    if not available:
        raise ValueError(f"No configured provider supports task '{task_type}'")

    if constraints.get("fastest"):
        return min(available, key=lambda p: PROVIDER_METRICS[p][task_type]["latency"])
    elif constraints.get("cheapest"):
        return min(available, key=lambda p: PROVIDER_METRICS[p][task_type]["cost"])
    else:
        # Default priority order
        priority = [ProviderName.OPENAI, ProviderName.ANTHROPIC, ProviderName.GROQ,
                    ProviderName.DEEPSEEK, ProviderName.HUGGINGFACE,
                    ProviderName.STABILITY, ProviderName.GROK]
        for p in priority:
            if p in available:
                return p
        return available[0]


async def _call(provider: ProviderName, task_type: str, input_text: str) -> str:
    """Call a provider using its configured env API key."""
    api_key = os.getenv(PROVIDER_KEY_ENV[provider], "")
    return await PROVIDER_CALLS[provider](task_type, input_text, api_key)

# ---------------------------------------------------------------------------
# Hybrid mode: call multiple providers in parallel, pick best result
# ---------------------------------------------------------------------------

async def hybrid_call(task_type: str, input_text: str) -> tuple:
    """
    Call OpenAI + Anthropic in parallel for text tasks.
    Simple quality heuristic: longer response wins (can be upgraded to ML scoring).
    Returns (result, providers_used, total_cost, latency_ms).
    """
    providers_to_use = []
    for p in [ProviderName.OPENAI, ProviderName.ANTHROPIC]:
        if PROVIDER_METRICS[p].get(task_type) and os.getenv(PROVIDER_KEY_ENV[p], ""):
            providers_to_use.append(p)

    if not providers_to_use:
        raise ValueError("No providers available for hybrid mode")

    start = time.time()
    tasks = [_call(p, task_type, input_text) for p in providers_to_use]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    latency = int((time.time() - start) * 1000)

    valid = [(p, r) for p, r in zip(providers_to_use, results) if isinstance(r, str)]
    if not valid:
        raise RuntimeError("All providers failed in hybrid mode")

    # Pick best: longest non-error response (simple heuristic for MVP)
    best_provider, best_result = max(valid, key=lambda x: len(x[1]))
    total_cost = sum(
        PROVIDER_METRICS[p][task_type]["cost"]
        for p, _ in valid
    )
    return best_result, [p.value for p, _ in valid], total_cost, latency

# ---------------------------------------------------------------------------
# Main orchestration entry point
# ---------------------------------------------------------------------------

async def orchestrate(
    task_type: str,
    input_text: str,
    constraints: dict,
) -> tuple:
    """
    Orchestrate an AI task.
    Returns: (provider_name, result, cost, latency_ms, status, hybrid, providers_used)
    """
    # --- Hybrid mode ---
    if constraints.get("hybrid") and task_type == "text":
        try:
            result, providers_used, cost, latency = await hybrid_call(task_type, input_text)
            return "hybrid", result, cost, latency, "success", True, providers_used
        except Exception as e:
            # Fall through to single-provider mode
            pass

    # --- Single provider mode with failover ---
    available = available_providers_for_task(task_type)
    if not available:
        return "none", None, 0.0, 0, "failed: no providers", False, []

    primary = choose_provider(task_type, constraints)
    # Build failover list: primary first, then rest sorted by latency
    others = [p for p in available if p != primary]
    others.sort(key=lambda p: PROVIDER_METRICS[p][task_type]["latency"])
    ordered = [primary] + others

    start = time.time()
    for i, provider in enumerate(ordered):
        try:
            result = await _call(provider, task_type, input_text)
            latency = int((time.time() - start) * 1000)
            cost = PROVIDER_METRICS[provider][task_type]["cost"]
            status = "success" if i == 0 else f"success (failover from {primary.value})"
            return provider.value, result, cost, latency, status, False, [provider.value]
        except Exception as e:
            if i == len(ordered) - 1:
                latency = int((time.time() - start) * 1000)
                return provider.value, None, 0.0, latency, f"failed: {str(e)[:100]}", False, []
            # Try next provider
            continue

    latency = int((time.time() - start) * 1000)
    return "none", None, 0.0, latency, "failed: all providers exhausted", False, []
