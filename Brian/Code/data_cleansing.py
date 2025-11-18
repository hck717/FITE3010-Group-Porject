# ==============================================================
#  FULL MERGE DEBUG + COMPARISON + CLEANING
#  - Uses spy_rth_volatility as master calendar
#  - Robust date column detection (no KeyErrors)
#  - Compares missing in original vs merged
#  - Fixes pct_change warning
#  - Prints everything
#  - Saves final Parquet + CSV
# ==============================================================

import pandas as pd
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_stdlib_context

# ------------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------------
CLOSE_COL = 'spy_ohlcv_1drth_close'

csv_urls = {
    "macro_positioning_data": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Nayoung/Data/SPY_with_macro_positioning.csv",
    "sentiment_data": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Brian/Data/daily_average_sentiment_score.csv",
    "sector_data": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Brian/Data/spy_sector_data_Brian.csv",
    "market_indicators": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Daniel/market_indicators.csv",
    "market_indicators_extended": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Daniel/market_indicators_extended.csv",
    "skew_hsitory": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Daniel/skew_history.csv",
    "vix3m_history": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Daniel/vix3m_history.csv",
    "vix_history": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Daniel/vix_history.csv",
    "vvix_history": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Daniel/vvix_history.csv",
    "spy_ohlcv_1drth": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Michael/Data/spy_ohlcv_1drth_20141231_20250602.csv",
    "spy_rth_gaps_overnight": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Michael/Data/spy_rth_gaps_overnight_20141231-20250602.csv",
    "spy_rth_indicators": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Michael/Data/spy_rth_indicators_20141231-20250602.csv",
    "spy_rth_trend_liquidity_pressure": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Michael/Data/spy_rth_trend_liquidity_pressure_20141231-20250602.csv",
    "spy_rth_trend_meanrev": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Michael/Data/spy_rth_trend_meanrev_20141231-20250602.csv",
    "spy_rth_volatility": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Michael/Data/spy_rth_volatility_20141231-20250602.csv",
    "spy_fx_flows": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Tee/Data/spy_fx_flows.csv"
}

# ------------------------------------------------------------------
# 2. PARSE DATES
# ------------------------------------------------------------------
def parse_dates(series):
    series = series.astype(str).str.strip()
    formats = ['%Y-%m-%d %H:%M:%S', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors='coerce')
        if parsed.notna().any():
            return parsed
    return pd.to_datetime(series, errors='coerce')

# ------------------------------------------------------------------
# 3. PROCESS + STORE ORIGINAL MISSING
# ------------------------------------------------------------------
processed = {}
original_missing = {}

for name, url in csv_urls.items():
    df_raw = pd.read_csv(url, engine="python", on_bad_lines="skip")
    print(f"\nLoaded {name}: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
    print(f"   Columns: {list(df_raw.columns)}")

    # Robust date col detection
    date_col = None
    lower_cols = {col.lower(): col for col in df_raw.columns}
    for pref in ['time', 'date', 'timestamp', 'quotedate']:
        if pref in lower_cols:
            date_col = lower_cols[pref]
            break
    if not date_col:
        print(f"⚠️ Skipping {name}, no date-like column found")
        continue

    # Parse + clean
    df = df_raw.copy()
    df['Date'] = parse_dates(df[date_col])
    df = df.dropna(subset=['Date']).copy()
    df['Date'] = df['Date'].dt.normalize().dt.tz_localize(None)

    # Prefix
    cols_to_prefix = [c for c in df.columns if c != 'Date']
    df = df.rename(columns={c: f"{name}_{c}" for c in cols_to_prefix})

    # Store original missing
    orig_miss = df_raw.drop(columns=[date_col], errors='ignore').isnull().sum()
    orig_miss_pct = (orig_miss / len(df_raw) * 100).round(2)
    original_missing[name] = orig_miss_pct[orig_miss_pct > 0]

    processed[name] = df
    print(f"  Valid rows: {len(df):,}")

# ------------------------------------------------------------------
# 4. MASTER CALENDAR = spy_rth_volatility
# ------------------------------------------------------------------
calendar = processed["spy_rth_volatility"][["Date"]].drop_duplicates().sort_values("Date").reset_index(drop=True)
merged = calendar.copy()
print(f"\nMASTER CALENDAR: {len(calendar):,} rows from spy_rth_volatility")

# ------------------------------------------------------------------
# 5. MERGE
# ------------------------------------------------------------------
for name, df in processed.items():
    if name == "spy_rth_volatility":
        continue  # already backbone
    merged = pd.merge(merged, df, on="Date", how="left")

merged = merged.sort_values("Date").reset_index(drop=True)
print(f"\nMERGED: {len(merged):,} rows × {len(merged.columns)} cols")

# ------------------------------------------------------------------
# 6. COMPARE MISSING
# ------------------------------------------------------------------
print(f"\n" + "="*80)
print(f"MISSING VALUE COMPARISON: ORIGINAL vs MERGED")
print(f"="*80)

for name in csv_urls.keys():
    if name not in processed:
        continue
    cols_in_merged = [c for c in merged.columns if c.startswith(name)]
    if not cols_in_merged:
        continue

    merged_miss = merged[cols_in_merged].isnull().sum()
    merged_miss_pct = (merged_miss / len(merged) * 100).round(2)

    orig_miss = original_missing.get(name, pd.Series(dtype=float))
    print(f"\n{name.upper()}")
    print(f"   Original rows: {len(processed[name]):,}")
    print(f"   Merged rows: {len(merged):,}")
    print(f"   Features: {len(cols_in_merged)}")

    if not orig_miss.empty:
        print(f"   ORIGINAL MISSING (%):")
        print(orig_miss.sort_values(ascending=False).head(5).to_string())
    else:
        print(f"   ORIGINAL: No missing values")

    print(f"   MERGED MISSING (%):")
    print(merged_miss_pct[merged_miss_pct > 0].sort_values(ascending=False).head(5).to_string())

# ------------------------------------------------------------------
# 7. CLEANING
# ------------------------------------------------------------------
print(f"\n" + "="*60)
print(f"CLEANING...")
print(f"="*60)

# Convert object cols
object_cols = merged.select_dtypes(include='object').columns
if len(object_cols) > 0:
    print(f"Converting {len(object_cols)} object columns to numeric...")
    for col in object_cols:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

# Drop 100% missing
miss_pct = merged.drop(columns='Date').isnull().mean()
cols_100 = miss_pct[miss_pct == 1].index
print(f"Dropping {len(cols_100)} columns (100% missing)")
merged = merged.drop(columns=cols_100)
# Create target
print(f"Creating SPY_NextDay_Return from {CLOSE_COL}...")
close = merged[CLOSE_COL].ffill()
returns = close.pct_change(fill_method=None).shift(-1)
merged['SPY_NextDay_Return'] = returns
merged = merged.dropna(subset=['SPY_NextDay_Return'])
print(f"Target created. Final rows: {len(merged):,}")

# Forward-fill macro columns
macro_cols = [c for c in merged.columns if 'macro_positioning_data' in c]
if macro_cols:
    merged[macro_cols] = merged[macro_cols].ffill()
    print(f"Forward-filled {len(macro_cols)} macro columns")

# ------------------------------------------------------------------
# 8. FINAL STATS
# ------------------------------------------------------------------
print(f"\n" + "="*60)
print(f"FINAL CLEAN DATASET")
print(f"="*60)
print(f"Rows: {len(merged):,}")
print(f"Cols: {len(merged.columns):,}")
print(f"Date range: {merged['Date'].min().date()} → {merged['Date'].max().date()}")

final_miss = merged.drop(columns=['Date', 'SPY_NextDay_Return']).isnull().mean() * 100
print(f"\nTop 10 missing % (after cleaning):")
print(final_miss.round(2).sort_values(ascending=False).head(10))

# ------------------------------------------------------------------
# 9. SAVE
# ------------------------------------------------------------------
parquet_file = "master_spy_clean_final.parquet"
csv_file = "master_spy_clean_final.csv"

# merged.to_parquet(parquet_file, index=False)
merged.to_csv(csv_file, index=False, header=True)

print(f"\nSAVED:")
print(f"  → {parquet_file} (Parquet, fast & small)")
print(f"  → {csv_file} (CSV, readable)")
