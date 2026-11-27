"""
Central configuration for the Pristyn Care Health analysis pipeline.
No paths are hardcoded elsewhere in the codebase — everything routes through here.
"""

from pathlib import Path

# Project root = two levels up from this file (src/config.py -> project root)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"

RAW_FILES = {
    "sales": DATA_RAW_DIR / "da_sales_transactions.csv",
    "customers": DATA_RAW_DIR / "da_customer_data.csv",
    "products": DATA_RAW_DIR / "da_product_catalogue.csv",
}

PROCESSED_FILES = {
    "sales_clean": DATA_PROCESSED_DIR / "sales_clean.csv",
    "customers_clean": DATA_PROCESSED_DIR / "customers_clean.csv",
    "baseline_metrics": DATA_PROCESSED_DIR / "baseline_metrics.csv",
}

# Ensure processed dir exists at import time (idempotent)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Checkpoint 3: healthcare data packs ---
RAW_FILES.update({
    "appointments": DATA_RAW_DIR / "da_healthcare_appointments.csv",
    "satisfaction": DATA_RAW_DIR / "sunrise_patient_satisfaction.csv",
    "resources": DATA_RAW_DIR / "sunrise_resource_utilisation.csv",
})
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
