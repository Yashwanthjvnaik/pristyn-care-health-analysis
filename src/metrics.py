"""
Baseline metric calculations for Checkpoint 2.

These serve as the reference point against which future weeks' pipeline runs
and the eventual funnel/segmentation analysis (Checkpoint 3) will be compared.
"""

from __future__ import annotations
import pandas as pd


def channel_conversion_metrics(sales: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Revenue, transaction count, and customer count by acquisition channel."""
    txn_by_channel = (
        sales.groupby("Channel")
        .agg(Transactions=("Transaction_ID", "count"), Revenue=("Net_Revenue", "sum"))
        .reset_index()
    )
    cust_by_channel = (
        customers.groupby("Acquisition_Channel")
        .agg(Customers=("Customer_ID", "count"), Churned=("Churned", "sum"))
        .reset_index()
        .rename(columns={"Acquisition_Channel": "Channel"})
    )
    merged = txn_by_channel.merge(cust_by_channel, on="Channel", how="outer")
    merged["Churn_Rate_Pct"] = (merged["Churned"] / merged["Customers"] * 100).round(2)
    merged["Revenue_Per_Customer"] = (merged["Revenue"] / merged["Customers"]).round(2)
    return merged.sort_values("Revenue", ascending=False).reset_index(drop=True)


def category_performance(sales: pd.DataFrame) -> pd.DataFrame:
    """Revenue and return rate by product category."""
    perf = (
        sales.groupby("Category")
        .agg(
            Revenue=("Net_Revenue", "sum"),
            Transactions=("Transaction_ID", "count"),
            Returns=("Return_Flag", "sum"),
        )
        .reset_index()
    )
    perf["Return_Rate_Pct"] = (perf["Returns"] / perf["Transactions"] * 100).round(2)
    return perf.sort_values("Revenue", ascending=False).reset_index(drop=True)


def customer_lifecycle_baseline(customers: pd.DataFrame) -> dict:
    """Headline lifecycle metrics: overall churn rate, avg orders, avg revenue/customer."""
    total = len(customers)
    churned = customers["Churned"].sum()
    return {
        "total_customers": int(total),
        "churned_customers": int(churned),
        "overall_churn_rate_pct": round(churned / total * 100, 2) if total else 0,
        "avg_orders_per_customer": round(customers["Total_Orders"].mean(), 2),
        "avg_revenue_per_customer": round(customers["Total_Revenue"].mean(), 2),
        "repeat_customer_rate_pct": round(
            (customers["Total_Orders"] > 1).sum() / total * 100, 2
        )
        if total
        else 0,
    }


def discount_return_correlation(sales: pd.DataFrame) -> float:
    """Pearson correlation between discount % and return flag (proxy for Q4)."""
    return round(sales["Discount_Pct"].corr(sales["Return_Flag"]), 3)


def build_baseline_summary(sales: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Flatten all baseline metrics into a single tidy DataFrame for export."""
    lifecycle = customer_lifecycle_baseline(customers)
    corr = discount_return_correlation(sales)

    rows = [{"metric": k, "value": v} for k, v in lifecycle.items()]
    rows.append({"metric": "discount_return_correlation", "value": corr})
    rows.append({"metric": "total_transactions", "value": len(sales)})
    rows.append({"metric": "total_net_revenue", "value": round(sales["Net_Revenue"].sum(), 2)})

    return pd.DataFrame(rows)
