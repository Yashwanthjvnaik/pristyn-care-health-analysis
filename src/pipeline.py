"""
Main pipeline entry point for Checkpoint 2 — Data pipeline + baseline metrics.

Run with:
    python -m src.pipeline

This loads the raw data packs, cleans them, validates referential integrity,
computes baseline metrics, and writes cleaned/processed CSVs to data/processed/.
The pipeline is repeatable: re-running it always reproduces the same outputs
from the same raw inputs (idempotent, no manual steps).
"""

from __future__ import annotations
import logging
import sys

from src.config import PROCESSED_FILES
from src.utils import load_raw, clean_sales, clean_customers, check_referential_integrity
from src.metrics import (
    channel_conversion_metrics,
    category_performance,
    build_baseline_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run() -> None:
    logger.info("Loading raw data packs...")
    raw = load_raw()

    logger.info("Cleaning sales transactions...")
    sales_clean = clean_sales(raw["sales"])
    logger.info("  %d rows retained (of %d raw)", len(sales_clean), len(raw["sales"]))

    logger.info("Cleaning customer master...")
    customers_clean = clean_customers(raw["customers"])
    logger.info("  %d rows retained (of %d raw)", len(customers_clean), len(raw["customers"]))

    logger.info("Checking referential integrity...")
    integrity = check_referential_integrity(sales_clean, customers_clean, raw["products"])
    if integrity["orphan_customer_ids"] or integrity["orphan_product_ids"]:
        logger.warning("Referential integrity issues found: %s", integrity)
    else:
        logger.info("  No orphan Customer_IDs or Product_IDs found.")

    logger.info("Computing baseline metrics...")
    channel_metrics = channel_conversion_metrics(sales_clean, customers_clean)
    category_metrics = category_performance(sales_clean)
    baseline_summary = build_baseline_summary(sales_clean, customers_clean)

    logger.info("Writing processed outputs to data/processed/ ...")
    sales_clean.to_csv(PROCESSED_FILES["sales_clean"], index=False)
    customers_clean.to_csv(PROCESSED_FILES["customers_clean"], index=False)
    baseline_summary.to_csv(PROCESSED_FILES["baseline_metrics"], index=False)
    channel_metrics.to_csv(
        PROCESSED_FILES["baseline_metrics"].parent / "channel_metrics.csv", index=False
    )
    category_metrics.to_csv(
        PROCESSED_FILES["baseline_metrics"].parent / "category_metrics.csv", index=False
    )

    logger.info("Pipeline complete.")
    print("\n=== BASELINE METRICS SUMMARY ===")
    print(baseline_summary.to_string(index=False))
    print("\n=== CHANNEL METRICS ===")
    print(channel_metrics.to_string(index=False))
    print("\n=== CATEGORY METRICS ===")
    print(category_metrics.to_string(index=False))


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:  # noqa: BLE001 - top-level entry point
        logger.error("Pipeline failed: %s", exc)
        sys.exit(1)
