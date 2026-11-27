"""Unit tests for cleaning and metric functions. Run with: pytest"""

import pandas as pd
import pytest
from src.utils import clean_sales, clean_customers, check_referential_integrity
from src.metrics import channel_conversion_metrics, customer_lifecycle_baseline


@pytest.fixture
def sample_sales():
    return pd.DataFrame(
        {
            "Transaction_ID": ["T1", "T2", "T2", "T3"],  # T2 duplicated
            "Date": ["2025-01-01", "2025-01-02", "2025-01-02", "2025-01-03"],
            "Customer_ID": ["C1", "C1", "C1", "C2"],
            "Product_ID": ["P1", "P1", "P1", "P2"],
            "Category": ["Beauty", "Beauty", "Beauty", "Home"],
            "Units": [2, 1, 1, -1],  # negative unit is invalid
            "Unit_Price": [100, 200, 200, 50],
            "Discount_Pct": [10, 0, 0, 5],
            "Channel": ["App", "Website", "Website", "Offline"],
            "City": ["Pune", "Delhi", "Delhi", "Mumbai"],
            "Return_Flag": [0, 1, 1, 0],
        }
    )


@pytest.fixture
def sample_customers():
    return pd.DataFrame(
        {
            "Customer_ID": ["C1", "C2"],
            "City": ["Pune", "Mumbai"],
            "Age_Band": ["18-24", "25-34"],
            "Gender": ["F", "M"],
            "Acquisition_Channel": ["App", "Offline"],
            "First_Purchase_Date": ["2025-01-01", "2025-01-03"],
            "Total_Orders": [3, 1],
            "Total_Revenue": [500, 47.5],
            "Last_Purchase_Date": ["2025-02-01", "2025-01-03"],
            "Churned": [0, 1],
        }
    )


def test_clean_sales_deduplicates(sample_sales):
    cleaned = clean_sales(sample_sales)
    assert cleaned["Transaction_ID"].duplicated().sum() == 0


def test_clean_sales_drops_invalid_units(sample_sales):
    cleaned = clean_sales(sample_sales)
    # T3 had Units = -1, should be dropped
    assert "T3" not in cleaned["Transaction_ID"].values


def test_clean_sales_computes_net_revenue(sample_sales):
    cleaned = clean_sales(sample_sales)
    row = cleaned[cleaned["Transaction_ID"] == "T1"].iloc[0]
    expected = 2 * 100 * (1 - 10 / 100)
    assert row["Net_Revenue"] == pytest.approx(expected)


def test_clean_customers_removes_duplicates(sample_customers):
    dup_df = pd.concat([sample_customers, sample_customers.iloc[[0]]], ignore_index=True)
    cleaned = clean_customers(dup_df)
    assert cleaned["Customer_ID"].duplicated().sum() == 0


def test_referential_integrity_no_orphans(sample_sales, sample_customers):
    products = pd.DataFrame({"Product_ID": ["P1", "P2"]})
    cleaned_sales = clean_sales(sample_sales)
    result = check_referential_integrity(cleaned_sales, sample_customers, products)
    assert result["orphan_customer_ids"] == 0
    assert result["orphan_product_ids"] == 0


def test_lifecycle_baseline_churn_rate(sample_customers):
    result = customer_lifecycle_baseline(sample_customers)
    assert result["total_customers"] == 2
    assert result["churned_customers"] == 1
    assert result["overall_churn_rate_pct"] == 50.0


def test_channel_conversion_metrics_shape(sample_sales, sample_customers):
    cleaned_sales = clean_sales(sample_sales)
    result = channel_conversion_metrics(cleaned_sales, sample_customers)
    assert "Channel" in result.columns
    assert "Churn_Rate_Pct" in result.columns
