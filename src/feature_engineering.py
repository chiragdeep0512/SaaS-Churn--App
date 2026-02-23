
import pandas as pd


def create_account_base(accounts, subs):

    today = pd.Timestamp.today()

    # Tenure
    accounts["tenure_days"] = (today - accounts["signup_date"]).dt.days

    # Subscription Aggregations
    subs_agg = subs.groupby("account_id").agg(
        total_subscriptions=("subscription_id", "count"),
        total_upgrades=("upgrade_flag", "sum"),
        total_downgrades=("downgrade_flag", "sum"),
        avg_mrr=("mrr_amount", "mean"),
        max_mrr=("mrr_amount", "max"),
        total_revenue=("arr_amount", "sum"),
    ).reset_index()

    df = accounts.merge(subs_agg, on="account_id", how="left")

    return df

def add_churn_label(df, churn):

    churn_flag = churn.groupby("account_id").size().reset_index(name="churned")

    churn_flag["churned"] = 1

    df = df.merge(churn_flag[["account_id", "churned"]],
                  on="account_id",
                  how="left")

    df["churned"] = df["churned"].fillna(0)

    return df

def add_usage_features(df, usage, subs):

    # Merge usage with subscriptions to get account_id
    usage = usage.merge(
        subs[["subscription_id", "account_id"]],
        on="subscription_id",
        how="left"
    )

    usage_agg = usage.groupby("account_id").agg(
        total_usage=("usage_count", "sum"),
        avg_usage_duration=("usage_duration_secs", "mean"),
        total_errors=("error_count", "sum"),
        feature_diversity=("feature_name", "nunique"),
        beta_usage_ratio=("is_beta_feature", "mean")
    ).reset_index()

    df = df.merge(usage_agg, on="account_id", how="left")

    return df

def add_support_features(df, tickets):

    support_agg = tickets.groupby("account_id").agg(
        total_tickets=("ticket_id", "count"),
        avg_resolution_time=("resolution_time_hours", "mean"),
        avg_first_response=("first_response_time_minutes", "mean"),
        avg_satisfaction=("satisfaction_score", "mean"),
        escalation_rate=("escalation_flag", "mean")
    ).reset_index()

    df = df.merge(support_agg, on="account_id", how="left")

    return df

def simulate_churn(df):
    """
    Generates a realistic 'churned' column based on usage, support, revenue and trial info.
    """
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler

    # Features to use
    features = ["total_usage", "avg_usage_duration", "total_errors",
                "avg_satisfaction", "total_tickets", "total_upgrades",
                "total_downgrades", "tenure_days", "is_trial", "escalation_rate"]

    # Fill NaNs with 0 first
    df[features] = df[features].fillna(0)

    # Scale numeric features 0-1
    scaler = MinMaxScaler()
    df_scaled = df[features].copy()
    numeric_features = df_scaled.select_dtypes(include=['float64', 'int64']).columns
    df_scaled[numeric_features] = scaler.fit_transform(df_scaled[numeric_features])

    # Assign weights
    weights = {
        "total_usage": -0.4,
        "avg_usage_duration": -0.2,
        "total_errors": 0.3,
        "avg_satisfaction": -0.5,
        "total_tickets": 0.3,
        "total_upgrades": -0.3,
        "total_downgrades": 0.2,
        "tenure_days": -0.2,
        "is_trial": 0.4,
        "escalation_rate": 0.3
    }

    # Calculate churn probability
    df["churn_prob"] = sum(df_scaled[f]*w for f, w in weights.items())
    df["churn_prob"] = np.clip(df["churn_prob"], 0, 1)

    # Sample churn
    np.random.seed(42)
    df["churned"] = np.random.binomial(1, df["churn_prob"])

    return df