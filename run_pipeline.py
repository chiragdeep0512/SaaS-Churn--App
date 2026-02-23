#from src.preprocessing import load_data, convert_dates
from src.preprocessing import convert_dates, load_data

from src.feature_engineering import (
    create_account_base,
    add_churn_label,
    add_usage_features,
    add_support_features
)

accounts, churn, usage, subs, tickets = load_data()
accounts, churn, usage, subs, tickets = convert_dates(accounts, churn, usage, subs, tickets)

df = create_account_base(accounts, subs)
df = add_churn_label(df, churn)
df = add_usage_features(df, usage, subs)
df = add_support_features(df, tickets)

print(df.head())
print(df.shape)
print(df.isna().mean() * 100)
print("---------Churn Distribution Check---------")
print(df['churned'].value_counts())
print(df['churned'].value_counts(normalize=True))

df.to_csv("data/processed/final_dataset.csv", index=False)

from src.feature_engineering import simulate_churn

df = simulate_churn(df)