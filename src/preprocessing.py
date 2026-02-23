import pandas as pd
import os


def load_data():
    base_path = os.path.dirname(os.path.dirname(__file__))

    accounts = pd.read_csv(os.path.join(base_path, 'data/raw/ravenstack_accounts.csv'))
    churn = pd.read_csv(os.path.join(base_path, "data/raw/ravenstack_churn_events.csv"))
    usage = pd.read_csv(os.path.join(base_path, "data/raw/ravenstack_feature_usage.csv"))
    subs = pd.read_csv(os.path.join(base_path, "data/raw/ravenstack_subscriptions.csv"))
    tickets = pd.read_csv(os.path.join(base_path, "data/raw/ravenstack_support_tickets.csv"))

    return accounts, churn, usage, subs, tickets


def convert_dates(accounts, churn, usage, subs, tickets):

    accounts['signup_date'] = pd.to_datetime(accounts['signup_date'], errors='coerce')

    churn['churn_date'] = pd.to_datetime(churn['churn_date'], errors='coerce')

    usage['usage_date'] = pd.to_datetime(usage['usage_date'], errors='coerce')

    subs['start_date'] = pd.to_datetime(subs['start_date'], errors='coerce')
    subs['end_date'] = pd.to_datetime(subs['end_date'], errors='coerce')

    tickets['submitted_at'] = pd.to_datetime(tickets['submitted_at'], errors='coerce')
    tickets['closed_at'] = pd.to_datetime(tickets['closed_at'], errors='coerce')

    return accounts, churn, usage, subs, tickets


if __name__ == "__main__":
    accounts, churn, usage, subs, tickets = load_data()
    accounts, churn, usage, subs, tickets = convert_dates(accounts, churn, usage, subs, tickets)

    print("\n accounts",accounts.info())
    print("\n churn",churn.info())
    print("\n usage",usage.info())
    print("\n subs",subs.info())
    print("\n tickets",tickets.info())