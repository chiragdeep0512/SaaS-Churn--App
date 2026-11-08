# SaaS Customer & Churn Analysis

Descriptive analysis of a synthetic SaaS dataset (RavenStack): customer segments,
usage, support and churn events, with data-quality checks and a Streamlit dashboard.

**Live app:** [paste your Streamlit link]

## Data
Five synthetic tables: accounts, subscriptions, feature usage, support tickets
and churn events: 33,100 records covering 500 accounts.
Source: [dataset link]. The data is synthetic, so findings illustrate the
analysis method, not real customer behaviour.

## What this project does
- Builds customer-segment, usage, support and churn-event views
- Checks IDs and dates, and reports exceptions
- Reconciles row counts across tables
- Finds 312 conflicts between account churn flags and event history
- Separates 600 churn events from 352 affected accounts
  (one account can have more than one churn event)

## Limitations
- Synthetic data; no real business conclusions
- This is a descriptive analysis, not a churn prediction model
- [add 1-2 coverage limits from your own documentation]

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Version note
[Keep only if true] An earlier version trained a churn prediction model on a
simulated churn label. That was removed because the label was derived from the
same features, which makes the evaluation circular.
