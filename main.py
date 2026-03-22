import os
import uuid
import logging
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware

from models import TaskRequest, TaskResponse
from supabase_client import supabase
from providers import orchestrate, PROVIDER_METRICS
from cache import make_cache_key, get_cached, set_cached
from utils import generate_api_key
from billing import (
    OMNI_PRICING,
    create_stripe_customer,
    check_balance,
    deduct_balance,
    get_balance_usd,
    create_topup_session,
    create_setup_session,
    auto_recharge,
    get_billing_history,
    get_payment_methods,
    handle_webhook,
    add_balance,
    AUTO_RECHARGE_THRESHOLD_USD,
)

app = FastAPI(title="Omni AI Orchestration API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("omni")

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://your-frontend.vercel.app")


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def get_api_key(x_api_key: str = Header(...)):
    res = supabase.table("api_keys").select("*").eq("key", x_api_key).execute()
    if not res.data:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return res.data[0]


async def get_developer(api_key_info=Depends(get_api_key)):
    developer_id = api_key_info["developer_id"]
    res = supabase.table("developers").select("*").eq("id", developer_id).execute()
    if not res.data:
        raise HTTPException(status_code=404, detail="Developer not found")
    return res.data[0], api_key_info


# ---------------------------------------------------------------------------
# Core AI task endpoint
# ---------------------------------------------------------------------------

@app.post("/ai-task", response_model=TaskResponse)
async def ai_task(request: TaskRequest, auth=Depends(get_developer)):
    developer, api_key_info = auth
    developer_id        = developer["id"]
    stripe_customer_id  = developer.get("stripe_customer_id")
    task_type           = request.taskType
    input_text          = request.input
    constraints         = request.constraints or {}

    # --- Balance check ---
    if stripe_customer_id:
        balance = get_balance_usd(stripe_customer_id)
        if balance <= AUTO_RECHARGE_THRESHOLD_USD:
            recharged = auto_recharge(stripe_customer_id)
            if recharged:
                logger.info(f"Auto-recharged {stripe_customer_id}")
                balance = get_balance_usd(stripe_customer_id)
        check_balance(stripe_customer_id, task_type)

    # --- Cache lookup ---
    cacheable = task_type in ("text", "embedding") and not constraints.get("hybrid")
    cache_key = make_cache_key(task_type, input_text, constraints) if cacheable else None
    if cache_key:
        cached = get_cached(cache_key)
        if cached:
            if stripe_customer_id:
                cost = OMNI_PRICING.get(task_type, 0.005) * 0.5
                deduct_balance(stripe_customer_id, cost, f"Cached {task_type} call")
            return TaskResponse(**cached, cached=True)

    # --- Orchestrate ---
    provider, result, cost, latency, status, is_hybrid, providers_used = await orchestrate(
        task_type, input_text, constraints
    )

    # --- Deduct from Stripe balance ---
    omni_price = OMNI_PRICING.get(task_type, 0.005)
    if stripe_customer_id and "success" in status:
        deduct_balance(stripe_customer_id, omni_price, f"Omni {task_type} via {provider}")

    response_data = dict(
        provider=provider,
        taskType=task_type,
        result=result if result else "Request failed",
        cost=omni_price,
        latency_ms=latency,
        status=status,
        cached=False,
        hybrid=is_hybrid,
        providers_used=providers_used,
    )

    if cacheable and cache_key and "success" in status:
        set_cached(cache_key, response_data)

    log_data = {
        "id":             str(uuid.uuid4()),
        "developer_id":   developer_id,
        "provider":       provider,
        "task_type":      task_type,
        "input":          input_text[:500],
        "output":         (result or "")[:1000],
        "cost":           omni_price,
        "provider_cost":  cost,
        "latency_ms":     latency,
        "status":         status,
        "hybrid":         is_hybrid,
        "providers_used": ",".join(providers_used),
        "created_at":     datetime.utcnow().isoformat(),
    }
    try:
        supabase.table("api_requests").insert(log_data).execute()
    except Exception as e:
        logger.warning(f"Failed to log: {e}")

    return TaskResponse(**response_data)


# ---------------------------------------------------------------------------
# Dashboard usage
# ---------------------------------------------------------------------------

@app.get("/dashboard/usage")
async def get_usage(auth=Depends(get_developer)):
    developer, _ = auth
    developer_id       = developer["id"]
    stripe_customer_id = developer.get("stripe_customer_id")
    thirty_days_ago    = (datetime.utcnow() - timedelta(days=30)).isoformat()

    res = (
        supabase.table("api_requests")
        .select("*")
        .eq("developer_id", developer_id)
        .gte("created_at", thirty_days_ago)
        .execute()
    )
    requests = res.data

    total_calls     = len(requests)
    total_spent     = sum(r["cost"] for r in requests)
    failed_requests = sum(1 for r in requests if "failed" in r.get("status", ""))
    cached_requests = sum(1 for r in requests if r.get("status", "").startswith("cached"))
    failure_rate    = round((failed_requests / total_calls * 100) if total_calls else 0, 2)

    cost_by_provider: dict = {}
    calls_by_provider: dict = {}
    latency_sums: dict = {}

    for r in requests:
        p = r["provider"]
        cost_by_provider[p]  = cost_by_provider.get(p, 0) + r["cost"]
        calls_by_provider[p] = calls_by_provider.get(p, 0) + 1
        latency_sums[p]      = latency_sums.get(p, 0) + (r.get("latency_ms") or 0)

    avg_latency_by_provider = {
        p: round(latency_sums[p] / calls_by_provider[p])
        for p in calls_by_provider
    }

    daily: dict = {}
    for r in requests:
        day = r["created_at"][:10]
        if day not in daily:
            daily[day] = {"requests": 0, "cost": 0.0}
        daily[day]["requests"] += 1
        daily[day]["cost"]     += r["cost"]

    daily_list = sorted(
        [{"date": d, "requests": v["requests"], "cost": round(v["cost"], 6)}
         for d, v in daily.items()],
        key=lambda x: x["date"],
    )

    task_breakdown: dict = {}
    for r in requests:
        t = r.get("task_type", "unknown")
        task_breakdown[t] = task_breakdown.get(t, 0) + 1

    current_balance = 0.0
    if stripe_customer_id:
        try:
            current_balance = get_balance_usd(stripe_customer_id)
        except Exception:
            pass

    return {
        "totalCalls":            total_calls,
        "totalSpent":            round(total_spent, 6),
        "currentBalance":        round(current_balance, 4),
        "failedRequests":        failed_requests,
        "cachedRequests":        cached_requests,
        "failureRate":           failure_rate,
        "costPerProvider":       cost_by_provider,
        "callsPerProvider":      calls_by_provider,
        "avgLatencyPerProvider": avg_latency_by_provider,
        "taskBreakdown":         task_breakdown,
        "dailyUsage":            daily_list,
    }


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@app.post("/auth/login-or-register")
async def login_or_register(data: dict):
    email        = data.get("email")
    auth_user_id = data.get("auth_user_id")
    if not email or not auth_user_id:
        raise HTTPException(status_code=400, detail="Email and auth_user_id required")
    existing = supabase.table("developers").select("*").eq("auth_user_id", auth_user_id).execute()
    if existing.data:
        developer = existing.data[0]
        key_res = supabase.table("api_keys").select("*").eq("developer_id", developer["id"]).execute()
        if key_res.data:
            return {"apiKey": key_res.data[0]["key"], "developerId": developer["id"], "new": False}
        api_key = generate_api_key()
        supabase.table("api_keys").insert({"key": api_key, "developer_id": developer["id"], "created_at": datetime.utcnow().isoformat()}).execute()
        return {"apiKey": api_key, "developerId": developer["id"], "new": False}
    existing_email = supabase.table("developers").select("*").eq("email", email).execute()
    if existing_email.data:
        developer = existing_email.data[0]
        update_data = {"auth_user_id": auth_user_id}
        if not developer.get("stripe_customer_id"):
            try:
                stripe_customer_id = create_stripe_customer(email, developer["id"])
                update_data["stripe_customer_id"] = stripe_customer_id
            except Exception as e:
                logger.warning(f"Stripe customer creation failed: {e}")
        supabase.table("developers").update(update_data).eq("id", developer["id"]).execute()
        key_res = supabase.table("api_keys").select("*").eq("developer_id", developer["id"]).execute()
        if key_res.data:
            return {"apiKey": key_res.data[0]["key"], "developerId": developer["id"], "new": False}
        api_key = generate_api_key()
        supabase.table("api_keys").insert({"key": api_key, "developer_id": developer["id"], "created_at": datetime.utcnow().isoformat()}).execute()
        return {"apiKey": api_key, "developerId": developer["id"], "new": False}
    developer_id = str(uuid.uuid4())
    stripe_customer_id = None
    try:
        stripe_customer_id = create_stripe_customer(email, developer_id)
    except Exception as e:
        logger.warning(f"Stripe customer creation failed: {e}")
    supabase.table("developers").insert({"id": developer_id, "email": email, "auth_user_id": auth_user_id, "stripe_customer_id": stripe_customer_id, "created_at": datetime.utcnow().isoformat()}).execute()
    api_key = generate_api_key()
    supabase.table("api_keys").insert({"key": api_key, "developer_id": developer_id, "created_at": datetime.utcnow().isoformat()}).execute()
    return {"apiKey": api_key, "developerId": developer_id, "new": True}


@app.post("/auth/register")
async def register(email: str):
    existing = supabase.table("developers").select("*").eq("email", email).execute()
    if existing.data:
        raise HTTPException(status_code=400, detail="Email already registered")

    developer_id = str(uuid.uuid4())

    stripe_customer_id = None
    try:
        stripe_customer_id = create_stripe_customer(email, developer_id)
    except Exception as e:
        logger.warning(f"Stripe customer creation failed: {e}")

    supabase.table("developers").insert({
        "id":                 developer_id,
        "email":              email,
        "stripe_customer_id": stripe_customer_id,
        "created_at":         datetime.utcnow().isoformat(),
    }).execute()

    api_key = generate_api_key()
    supabase.table("api_keys").insert({
        "key":          api_key,
        "developer_id": developer_id,
        "created_at":   datetime.utcnow().isoformat(),
    }).execute()

    return {
        "apiKey":           api_key,
        "developerId":      developer_id,
        "stripeCustomerId": stripe_customer_id,
    }


@app.post("/auth/regenerate-key")
async def regenerate_key(auth=Depends(get_developer)):
    developer, api_key_info = auth
    old_key      = api_key_info["key"]
    developer_id = developer["id"]

    supabase.table("api_keys").delete().eq("key", old_key).execute()
    new_key = generate_api_key()
    supabase.table("api_keys").insert({
        "key":          new_key,
        "developer_id": developer_id,
        "created_at":   datetime.utcnow().isoformat(),
    }).execute()
    return {"apiKey": new_key}


@app.get("/auth/key")
async def get_key_info(auth=Depends(get_developer)):
    developer, api_key_info = auth
    return {
        "key":          api_key_info["key"],
        "created_at":   api_key_info["created_at"],
        "developer_id": developer["id"],
        "email":        developer["email"],
    }


# ---------------------------------------------------------------------------
# Billing endpoints
# ---------------------------------------------------------------------------

@app.get("/billing/balance")
async def get_balance_endpoint(auth=Depends(get_developer)):
    developer, _ = auth
    stripe_customer_id = developer.get("stripe_customer_id")
    if not stripe_customer_id:
        try:
            stripe_customer_id = create_stripe_customer(developer["email"], developer["id"])
            supabase.table("developers").update({"stripe_customer_id": stripe_customer_id}).eq("id", developer["id"]).execute()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not create billing account: {str(e)}")
    balance = get_balance_usd(stripe_customer_id)
    return {
        "balance_usd":                   round(balance, 4),
        "auto_recharge_threshold_usd":   AUTO_RECHARGE_THRESHOLD_USD,
    }


@app.post("/billing/topup")
async def topup(amount_usd: float, auth=Depends(get_developer)):
    developer, _ = auth
    stripe_customer_id = developer.get("stripe_customer_id")
    if not stripe_customer_id:
        try:
            stripe_customer_id = create_stripe_customer(developer["email"], developer["id"])
            supabase.table("developers").update({"stripe_customer_id": stripe_customer_id}).eq("id", developer["id"]).execute()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not create billing account: {str(e)}")
    checkout_url = create_topup_session(
        stripe_customer_id=stripe_customer_id,
        amount_usd=amount_usd,
        success_url=f"{FRONTEND_URL}/billing/success?amount={amount_usd}",
        cancel_url=f"{FRONTEND_URL}/billing",
    )
    return {"checkout_url": checkout_url}


@app.post("/billing/setup-card")
async def setup_card(auth=Depends(get_developer)):
    developer, _ = auth
    stripe_customer_id = developer.get("stripe_customer_id")
    if not stripe_customer_id:
        try:
            stripe_customer_id = create_stripe_customer(developer["email"], developer["id"])
            supabase.table("developers").update({"stripe_customer_id": stripe_customer_id}).eq("id", developer["id"]).execute()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not create billing account: {str(e)}")
    setup_url = create_setup_session(
        stripe_customer_id=stripe_customer_id,
        success_url=f"{FRONTEND_URL}/billing/card-saved",
        cancel_url=f"{FRONTEND_URL}/billing",
    )
    return {"setup_url": setup_url}


@app.get("/billing/history")
async def billing_history(auth=Depends(get_developer)):
    developer, _ = auth
    stripe_customer_id = developer.get("stripe_customer_id")
    if not stripe_customer_id:
        return {"transactions": []}
    history = get_billing_history(stripe_customer_id)
    return {"transactions": history}


@app.get("/billing/payment-methods")
async def payment_methods_endpoint(auth=Depends(get_developer)):
    developer, _ = auth
    stripe_customer_id = developer.get("stripe_customer_id")
    if not stripe_customer_id:
        return {"payment_methods": []}
    methods = get_payment_methods(stripe_customer_id)
    return {"payment_methods": methods}


@app.get("/billing/pricing")
async def get_pricing():
    return {
        "pricing":                       OMNI_PRICING,
        "currency":                      "usd",
        "minimum_topup_usd":             10.00,
        "auto_recharge_threshold_usd":   AUTO_RECHARGE_THRESHOLD_USD,
        "note":                          "Pay only for what you use. No monthly fees.",
    }


# ---------------------------------------------------------------------------
# Stripe webhook
# ---------------------------------------------------------------------------

@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    payload    = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    event      = handle_webhook(payload, sig_header)

    if event["type"] == "checkout.session.completed":
        session  = event["data"]["object"]
        metadata = session.get("metadata", {})
        if metadata.get("type") == "topup":
            stripe_customer_id = metadata.get("stripe_customer_id")
            amount_usd         = float(metadata.get("credit_amount_usd", 0))
            if stripe_customer_id and amount_usd > 0:
                add_balance(stripe_customer_id, amount_usd, f"Top-up ${amount_usd}")
                logger.info(f"Added ${amount_usd} credit to {stripe_customer_id}")

    elif event["type"] == "payment_intent.payment_failed":
        pi = event["data"]["object"]
        logger.warning(f"Payment failed for {pi.get('customer')}")

    return {"received": True}


# ---------------------------------------------------------------------------
# Providers + health
# ---------------------------------------------------------------------------

@app.get("/providers")
async def list_providers():
    out = []
    for provider, caps in PROVIDER_METRICS.items():
        supported = [task for task, metrics in caps.items() if metrics is not None]
        out.append({
            "name":           provider.value,
            "supportedTasks": supported,
            "metrics":        {t: caps[t] for t in supported},
        })
    return {"providers": out}


@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}
