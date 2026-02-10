Hibuddy store count scraper (v17 + speed/reliability fixes)

What's new in this build
- Gateway/Cloudflare recovery: retries + reload/back + return to /products/all.
- Faster scraping: blocks images/media/fonts AFTER you finish interactive setup.
- Candidate verification: still checks top 5 by default, but stops early if a match score >= 60 (configurable).

Install
1) python -m venv .venv
2) .\.venv\Scripts\activate   (Windows PowerShell)
3) pip install -r requirements.txt
4) playwright install chromium

Run
python hibuddy_storecount_scraper.py --input "assortment_shaped.xlsx" --interactive-setup

python hibuddy_storecount_scraper_copy.py \
  --input assortment_ON.xlsx \
  --output hibuddy_storecounts.xlsx \
  --profile-dir hibuddy_profile \
  --parallel --workers 5 \
  --interactive-setup


Notes
- Use the interactive setup once to set Location=Calgary, Alberta, Canada and Radius=50 km.
- After you press Enter, the script switches to speed mode and blocks images/media/fonts.
