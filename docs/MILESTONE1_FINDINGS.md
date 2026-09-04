# Milestone 1 — Data Inventory Findings (Summary)

Full report: `Milestone1_DataInventory.pdf` in this folder.

## Data Inventory

| File | Rows | Cols | Missing | Notes |
|---|---|---|---|---|
| da_sales_transactions.csv | 1,000 | 11 | 0 | Transaction-level: units, price, discount, channel, returns |
| da_customer_data.csv | 200 | 10 | 0 | Customer master: channel, orders, revenue, churn flag |
| da_product_catalogue.csv | 50 | 7 | 0 | Product master: MRP, COGS, stock, avg. monthly sales |

Referential integrity is clean — every `Customer_ID` and `Product_ID` in the
transactions file matches its master table. Zero duplicate `Transaction_ID`s.

## Critical Finding

The three provided files are generic multi-category retail transaction data.
None contain fields for the 7-stage patient funnel described in the project
brief (ad click → enquiry → consultation → diagnostics → booking → surgery →
follow-up). There is no top-of-funnel data (impressions, clicks, enquiries) —
only completed transactions and existing customers.

**Status:** escalated to stakeholder/mentor for clarification on whether
funnel-stage data exists separately, or whether the funnel should be reframed
around available signals. This pipeline (Checkpoint 2) proceeds with the
retention/revenue proxy in the meantime, per the recommendation in the full
Milestone 1 report.
