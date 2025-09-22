
# Truckload Analytics — No‑Code Snowflake SQL Builder (POC)

This app is **self-contained**: metadata is baked into `hardcoded_metadata.py`. End users do **not** upload CSVs.

## Files
- `app.py` — Streamlit app (uses baked-in metadata)
- `hardcoded_metadata.py` — auto-generated from your 4 CSVs (this repo has the final version)
- `requirements.txt`

## Deploy on Streamlit Cloud
1. Create a new app from this repo, entrypoint `app.py`.
2. Done. No CSVs required at runtime.

---

## Summary of baked metadata
- Source CSVs found: TRUCKLOAD_ETA_ANALYTICS(Akshat).csv, TRUCKLOAD_ETA_ANALYTICS(Anmol).csv, TRUCKLOAD_ETA_ANALYTICS(Raghav).csv, TRUCKLOAD_ETA_ANALYTICS(Sarthak).csv
- Missing CSVs (ignored): None
- Total tables baked: 102
