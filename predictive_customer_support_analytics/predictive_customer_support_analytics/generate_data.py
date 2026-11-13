"""Generate a synthetic customer support ticket dataset.

Two targets are created:
  - escalated        (1 = ticket was escalated to a senior agent)
  - resolution_hours (time taken to resolve the ticket)
"""
import numpy as np
import pandas as pd


def generate(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "channel": rng.choice(["email", "chat", "phone", "social"], n, p=[0.35, 0.30, 0.20, 0.15]),
        "category": rng.choice(["billing", "technical", "account", "shipping", "general"], n,
                               p=[0.20, 0.35, 0.15, 0.15, 0.15]),
        "priority": rng.choice(["low", "medium", "high", "urgent"], n, p=[0.35, 0.40, 0.18, 0.07]),
        "customer_tier": rng.choice(["free", "standard", "premium"], n, p=[0.40, 0.45, 0.15]),
        "previous_tickets": rng.poisson(2, n),
        "message_length": rng.integers(20, 1500, n),
        "sentiment_score": np.clip(rng.normal(0, 0.5, n), -1, 1).round(3),
        "first_response_minutes": rng.exponential(60, n).round(1),
        "account_age_days": rng.integers(10, 2000, n),
    })

    priority_w = df["priority"].map({"low": 0, "medium": 0.4, "high": 1.0, "urgent": 1.8})
    category_w = df["category"].map({"billing": 0.3, "technical": 0.8, "account": 0.2,
                                     "shipping": 0.3, "general": 0.0})
    tier_w = df["customer_tier"].map({"free": 0.0, "standard": 0.1, "premium": 0.3})

    # Escalation: driven by priority, negative sentiment, repeat contacts, slow first response
    logit = (-2.6
             + priority_w
             + category_w
             - 1.5 * df["sentiment_score"]
             + 0.15 * df["previous_tickets"]
             + 0.008 * df["first_response_minutes"]
             + tier_w
             + rng.normal(0, 0.3, n))
    p_escalate = 1 / (1 + np.exp(-logit))
    df["escalated"] = (rng.random(n) < p_escalate).astype(int)

    # Resolution time (hours)
    hours = (4
             + 6 * category_w
             + 5 * priority_w * 0.5
             + 0.004 * df["message_length"]
             + 0.02 * df["first_response_minutes"]
             + 12 * df["escalated"]
             - 1.5 * df["sentiment_score"]
             + rng.normal(0, 2, n))
    df["resolution_hours"] = np.clip(hours, 0.5, None).round(2)

    return df


if __name__ == "__main__":
    data = generate()
    data.to_csv("tickets.csv", index=False)
    print(f"Saved tickets.csv with {len(data)} rows")
    print(f"Escalation rate: {data['escalated'].mean():.1%}")
