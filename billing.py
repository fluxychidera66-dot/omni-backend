"""
billing.py — Stripe pay-as-you-go billing for Omni API

Flow:
1. Developer registers → Stripe customer created automatically
2. Developer tops up via Stripe Checkout → balance added to their account
3. Every API call → balance checked and cost deducted
4. Balance runs low → auto recharge from saved card
5. Stripe webhooks → keep everything in sync
"""

import os
import stripe
from fastapi import HTTPException

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# ---------------------------------------------------------------------------
# Pricing — what Omni charges customers per call (USD)
# ---------------------------------------------------------------------------

OMNI_PRICING = {
    "text":             0.005,   # $0.005 per call
    "embedding":        0.001,   # $0.001 per call
    "image":            0.05,    # $0.05  per image
    "video":            0.50,    # $0.50  per video
    "text-to-speech":   0.01,    # $0.01  per call
    "speech-to-text":   0.01,    # $0.01  per minute
}

MINIMUM_TOPUP_USD          = 10.0   # Minimum top-up amount
AUTO_RECHARGE_THRESHOLD_USD = 0.02  # Auto-recharge when balance drops below $0.02
AUTO_RECHARGE_AMOUNT_USD    = 10.0  # Amount to auto-recharge


# ---------------------------------------------------------------------------
# Customer management
# ---------------------------------------------------------------------------

def create_stripe_customer(email: str, developer_id: str) -> str:
    """Create a Stripe customer. Returns Stripe customer ID."""
    customer = stripe.Customer.create(
        email=email,
        metadata={
            "developer_id":  developer_id,
            "balance_usd":   "0.0",
        },
    )
    return customer.id


# ---------------------------------------------------------------------------
# Balance management (stored in Stripe customer metadata)
# ---------------------------------------------------------------------------

def get_balance_usd(stripe_customer_id: str) -> float:
    """Get current credit balance in USD."""
    customer = stripe.Customer.retrieve(stripe_customer_id)
    return float(customer.metadata.get("balance_usd", "0.0"))


def set_balance_usd(stripe_customer_id: str, balance: float):
    """Set credit balance in USD."""
    stripe.Customer.modify(
        stripe_customer_id,
        metadata={"balance_usd": str(round(balance, 6))},
    )


def add_balance(stripe_customer_id: str, amount_usd: float, description: str = "Top-up"):
    """Add credit to customer balance."""
    current = get_balance_usd(stripe_customer_id)
    set_balance_usd(stripe_customer_id, current + amount_usd)


def deduct_balance(stripe_customer_id: str, amount_usd: float, description: str = "API call"):
    """Deduct cost from balance. Raises 402 if insufficient funds."""
    current = get_balance_usd(stripe_customer_id)
    if current < amount_usd:
        raise HTTPException(
            status_code=402,
            detail={
                "error":                "insufficient_credits",
                "message":              "Your Omni credit balance is too low. Please top up to continue.",
                "current_balance_usd":  round(current, 4),
                "required_usd":         round(amount_usd, 4),
                "topup_url":            "/billing/topup",
            }
        )
    set_balance_usd(stripe_customer_id, current - amount_usd)


def check_balance(stripe_customer_id: str, task_type: str):
    """
    Check if customer has enough balance for a task.
    Raises 402 if not enough funds.
    """
    cost    = OMNI_PRICING.get(task_type, 0.005)
    balance = get_balance_usd(stripe_customer_id)
    if balance < cost:
        raise HTTPException(
            status_code=402,
            detail={
                "error":                "insufficient_credits",
                "message":              f"Insufficient credits. This task costs ${cost}. Your balance is ${round(balance, 4)}.",
                "current_balance_usd":  round(balance, 4),
                "required_usd":         cost,
                "topup_url":            "/billing/topup",
            }
        )


# ---------------------------------------------------------------------------
# Auto-recharge
# ---------------------------------------------------------------------------

def auto_recharge(stripe_customer_id: str) -> bool:
    """
    Charge customer's default payment method automatically.
    Returns True if successful, False otherwise.
    """
    try:
        customer   = stripe.Customer.retrieve(stripe_customer_id)
        default_pm = customer.get("invoice_settings", {}).get("default_payment_method")
        if not default_pm:
            return False

        amount_cents = int(AUTO_RECHARGE_AMOUNT_USD * 100)
        intent = stripe.PaymentIntent.create(
            amount=amount_cents,
            currency="usd",
            customer=stripe_customer_id,
            payment_method=default_pm,
            confirm=True,
            off_session=True,
            metadata={
                "type":                "omni_auto_recharge",
                "stripe_customer_id":  stripe_customer_id,
                "credit_amount_usd":   str(AUTO_RECHARGE_AMOUNT_USD),
            },
            description="Omni API auto-recharge",
        )

        if intent.status == "succeeded":
            add_balance(stripe_customer_id, AUTO_RECHARGE_AMOUNT_USD, "Auto-recharge")
            return True
        return False

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Checkout sessions (Stripe hosted pages — no card handling in your code)
# ---------------------------------------------------------------------------

def create_topup_session(
    stripe_customer_id: str,
    amount_usd: float,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Create a Stripe Checkout session for credit top-up.
    Returns the hosted checkout URL to redirect the customer to.
    """
    if amount_usd < MINIMUM_TOPUP_USD:
        raise HTTPException(
            status_code=400,
            detail=f"Minimum top-up is ${MINIMUM_TOPUP_USD:.0f}"
        )

    amount_cents = int(amount_usd * 100)

    session = stripe.checkout.Session.create(
        customer=stripe_customer_id,
        payment_method_types=["card"],
        line_items=[{
            "price_data": {
                "currency":     "usd",
                "unit_amount":  amount_cents,
                "product_data": {
                    "name":        "Omni API Credits",
                    "description": f"${amount_usd:.2f} credit for Omni AI API usage",
                },
            },
            "quantity": 1,
        }],
        mode="payment",
        success_url=success_url,
        cancel_url=cancel_url,
        metadata={
            "type":                "topup",
            "stripe_customer_id":  stripe_customer_id,
            "credit_amount_usd":   str(amount_usd),
        },
    )
    return session.url


def create_setup_session(
    stripe_customer_id: str,
    success_url: str,
    cancel_url: str,
) -> str:
    """
    Create a Stripe SetupIntent session to save a card for auto-recharge.
    Returns the hosted setup URL.
    """
    session = stripe.checkout.Session.create(
        customer=stripe_customer_id,
        payment_method_types=["card"],
        mode="setup",
        success_url=success_url,
        cancel_url=cancel_url,
    )
    return session.url


# ---------------------------------------------------------------------------
# Billing history & payment methods
# ---------------------------------------------------------------------------

def get_billing_history(stripe_customer_id: str) -> list:
    """Return list of successful payments."""
    intents = stripe.PaymentIntent.list(customer=stripe_customer_id, limit=20)
    history = []
    for intent in intents.data:
        if intent.status == "succeeded":
            history.append({
                "id":          intent.id,
                "amount_usd":  intent.amount / 100,
                "date":        intent.created,
                "type":        intent.metadata.get("type", "topup"),
                "description": intent.description or "Payment",
            })
    return history


def get_payment_methods(stripe_customer_id: str) -> list:
    """Return saved payment methods for a customer."""
    methods = stripe.PaymentMethod.list(
        customer=stripe_customer_id,
        type="card",
    )
    result = []
    for pm in methods.data:
        card = pm.card
        result.append({
            "id":        pm.id,
            "brand":     card.brand,
            "last4":     card.last4,
            "exp_month": card.exp_month,
            "exp_year":  card.exp_year,
        })
    return result


# ---------------------------------------------------------------------------
# Webhook handler
# ---------------------------------------------------------------------------

def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """Verify Stripe webhook signature and return event dict."""
    webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    try:
        event = stripe.Webhook.construct_event(payload, sig_header, webhook_secret)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    return event
