# Deployment / Environment Notes

This checkpoint runs entirely locally — there is no hosted app or service yet.

## Environment

- Python 3.11+
- Dependencies pinned in `requirements.txt`

## Environment Variables

None required for this checkpoint. If credentials or external endpoints are
introduced in a later checkpoint (e.g. a hosted dashboard), they will be
documented here and loaded via `.env` (never committed — see `.gitignore`).

## Known Issues

- Churn is under-represented in the sample (3%, 6/200 customers), which is
  low for a healthcare/elective-service context. Flagged in Milestone 1;
  treat churn-related conclusions as directional, not final, until validated
  against a larger sample.
- Product catalogue only covers 5 of the 8 categories present in the
  transactions data (missing Sports, Books, Health) — so margin/MRP-based
  analysis is incomplete for those three categories.

## Reproducing Results

```bash
git clone <this-repo-url>
cd pristyn-care-health-analysis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline
pytest tests/ -v
```
