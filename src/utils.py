"""
Shared utilities: loading, cleaning, and validating the Pristyn Care Health
retail dataset (sales transactions, customer master, product catalogue).

Note on scope: the source data is generic multi-category retail data and does
not contain patient-funnel fields (ad click / enquiry / consultation / etc.)
as originally described in the project brief — see Milestone 1 data inventory.
This pipeline therefore builds a post-purchase retention & revenue baseline
(acquisition -> first purchase -> repeat orders -> churn) pending stakeholder
confirmation of true funnel-stage data.
"""

from __future__ import annotations
import pandas as pd
from src.config import RAW_FILES


def load_raw() -> dict[str, pd.DataFrame]:
    """Load the three raw CSVs into DataFrames, keyed by name."""
    return {
        "sales": pd.read_csv(RAW_FILES["sales"], parse_dates=["Date"]),
        "customers": pd.read_csv(
            RAW_FILES["customers"],
            parse_dates=["First_Purchase_Date", "Last_Purchase_Date"],
        ),
        "products": pd.read_csv(RAW_FILES["products"]),
    }


def clean_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate, validate types, and derive revenue for the sales table."""
    df = sales.drop_duplicates(subset="Transaction_ID").copy()

    numeric_cols = ["Units", "Unit_Price", "Discount_Pct", "Return_Flag"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that failed numeric coercion or have non-physical values
    df = df.dropna(subset=numeric_cols)
    df = df[(df["Units"] > 0) & (df["Unit_Price"] > 0)]
    df = df[(df["Discount_Pct"] >= 0) & (df["Discount_Pct"] <= 100)]

    df["Net_Revenue"] = (
        df["Units"] * df["Unit_Price"] * (1 - df["Discount_Pct"] / 100)
    ).round(2)

    return df.reset_index(drop=True)


def clean_customers(customers: pd.DataFrame) -> pd.DataFrame:
    """Validate types and flag inconsistent customer records."""
    df = customers.drop_duplicates(subset="Customer_ID").copy()
    df["Total_Orders"] = pd.to_numeric(df["Total_Orders"], errors="coerce")
    df["Total_Revenue"] = pd.to_numeric(df["Total_Revenue"], errors="coerce")
    df["Churned"] = pd.to_numeric(df["Churned"], errors="coerce").fillna(0).astype(int)
    df = df.dropna(subset=["Total_Orders", "Total_Revenue"])
    return df.reset_index(drop=True)


def check_referential_integrity(
    sales: pd.DataFrame, customers: pd.DataFrame, products: pd.DataFrame
) -> dict[str, int]:
    """Return counts of orphan Customer_IDs / Product_IDs found in sales."""
    orphan_customers = set(sales["Customer_ID"]) - set(customers["Customer_ID"])
    orphan_products = set(sales["Product_ID"]) - set(products["Product_ID"])
    return {
        "orphan_customer_ids": len(orphan_customers),
        "orphan_product_ids": len(orphan_products),
    }
