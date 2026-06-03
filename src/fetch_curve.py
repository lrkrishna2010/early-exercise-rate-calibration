"""
fetch_curve.py -- download the current U.S. Treasury par-yield curve.

Run locally (Treasury/FRED are not reachable from restricted sandboxes):
    pip install pandas requests
    python fetch_curve.py

Prints the CMT par yields you can paste into rate_data.py (TREASURY_TENORS /
TREASURY_PAR_CURVE). Treasury publishes the daily par yield curve as XML/CSV at
home.treasury.gov; this script pulls the latest row.
"""
import sys
import pandas as pd

URL = ("https://home.treasury.gov/resource-center/data-chart-center/"
       "interest-rates/daily-treasury-rates.csv/{year}/all?"
       "type=daily_treasury_yield_curve&field_tdr_date_value={year}&page&_format=csv")

TENOR_MAP = {"1 Mo": 0.0833, "3 Mo": 0.25, "6 Mo": 0.5, "1 Yr": 1.0, "2 Yr": 2.0,
             "3 Yr": 3.0, "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0,
             "20 Yr": 20.0, "30 Yr": 30.0}

def main():
    import datetime
    year = datetime.date.today().year
    df = pd.read_csv(URL.format(year=year))
    latest = df.iloc[0]
    tenors, yields = [], []
    for col, t in TENOR_MAP.items():
        if col in latest and pd.notna(latest[col]):
            tenors.append(t); yields.append(round(float(latest[col]) / 100, 4))
    print("# as of", latest["Date"])
    print("TREASURY_TENORS =", tenors)
    print("TREASURY_PAR_CURVE =", yields)

if __name__ == "__main__":
    main()
