# ==============================================================
#  FULL MERGE DEBUG + COMPARISON + CLEANING + FEATURE SELECTION
#  - Uses spy_rth_volatility as master calendar
#  - Robust date column detection (no KeyErrors)
#  - Compares missing in original vs merged
#  - Fixes pct_change warning
#  - Prints everything
#  - Creates target (SPY_NextDay_Return)
#  - Strict Feature Selection + Correlation Pruning (TRAIN ONLY)
#  - Saves final Parquet + CSV + final_feature_list.csv
# ==============================================================

import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. CONFIG
# ------------------------------------------------------------------
CLOSE_COL = "spy_ohlcv_1drth_close"

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
    "spy_fx_flows": "https://raw.githubusercontent.com/hck717/FITE3010-Group-Porject/main/Tee/Data/spy_fx_flows.csv",
}

# ------------------------------------------------------------------
# 2. PARSE DATES
# ------------------------------------------------------------------
def parse_dates(series: pd.Series) -> pd.Series:
    series = series.astype(str).str.strip()
    formats = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        if parsed.notna().any():
            return parsed
    return pd.to_datetime(series, errors="coerce")


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
    for pref in ["time", "date", "timestamp", "quotedate"]:
        if pref in lower_cols:
            date_col = lower_cols[pref]
            break
    if not date_col:
        print(f"⚠️ Skipping {name}, no date-like column found")
        continue

    # Parse + clean
    df = df_raw.copy()
    df["Date"] = parse_dates(df[date_col])
    df = df.dropna(subset=["Date"]).copy()
    df["Date"] = df["Date"].dt.normalize().dt.tz_localize(None)

    # Prefix
    cols_to_prefix = [c for c in df.columns if c != "Date"]
    df = df.rename(columns={c: f"{name}_{c}" for c in cols_to_prefix})

    # Store original missing (from raw)
    orig_miss = df_raw.drop(columns=[date_col], errors="ignore").isnull().sum()
    orig_miss_pct = (orig_miss / len(df_raw) * 100).round(2)
    original_missing[name] = orig_miss_pct[orig_miss_pct > 0]

    processed[name] = df
    print(f"  Valid rows: {len(df):,}")

# ------------------------------------------------------------------
# 4. MASTER CALENDAR = spy_rth_volatility
# ------------------------------------------------------------------
calendar = (
    processed["spy_rth_volatility"][["Date"]]
    .drop_duplicates()
    .sort_values("Date")
    .reset_index(drop=True)
)
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
print("\n" + "=" * 80)
print("MISSING VALUE COMPARISON: ORIGINAL vs MERGED")
print("=" * 80)

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
        print("   ORIGINAL MISSING (%):")
        print(orig_miss.sort_values(ascending=False).head(5).to_string())
    else:
        print("   ORIGINAL: No missing values")

    print("   MERGED MISSING (%):")
    print(
        merged_miss_pct[merged_miss_pct > 0]
        .sort_values(ascending=False)
        .head(5)
        .to_string()
    )

# ------------------------------------------------------------------
# 7. CLEANING
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("CLEANING...")
print("=" * 60)

# Convert object cols
object_cols = merged.select_dtypes(include="object").columns
if len(object_cols) > 0:
    print(f"Converting {len(object_cols)} object columns to numeric...")
    for col in object_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

# Drop 100% missing
miss_pct = merged.drop(columns="Date").isnull().mean()
cols_100 = miss_pct[miss_pct == 1].index
print(f"Dropping {len(cols_100)} columns (100% missing)")
merged = merged.drop(columns=cols_100)

# Create target (next-day return) safely
print(f"Creating SPY_NextDay_Return from {CLOSE_COL}...")
close = merged[CLOSE_COL].ffill()
returns = close.pct_change(fill_method=None).shift(-1)
merged["SPY_NextDay_Return"] = returns
merged = merged.dropna(subset=["SPY_NextDay_Return"])
print(f"Target created. Final rows: {len(merged):,}")

# Forward-fill macro columns
macro_cols = [c for c in merged.columns if "macro_positioning_data" in c]
if macro_cols:
    merged[macro_cols] = merged[macro_cols].ffill()
    print(f"Forward-filled {len(macro_cols)} macro columns")

# ------------------------------------------------------------------
# 8. STRICT FEATURE SELECTION + CORRELATION PRUNING (ANTI-LEAKAGE)
#     IMPORTANT: correlation audit MUST be done on TRAIN only.
# ------------------------------------------------------------------

# ----------------------------- Strict Feature Selection -----------------------------
forbidden_features = [
    "SPY_Close",
    "SPY_Close_Next",
    "price_change",
    "vol_price",
    "target_price_z",
    "SPY_NextDay_Return",
]

forbidden_keywords = ["AdjClose", "Open", "High", "Low", "Close"]
sector_price_prefix = "sector_data_Prices"
macro_raw = macro_cols

features = []
for col in merged.columns:
    if col in ["Date", "SPY_NextDay_Return"]:
        continue
    if col in forbidden_features:
        continue
    if any(keyword in col for keyword in forbidden_keywords):
        continue
    if col.startswith(sector_price_prefix):
        continue
    if col in macro_raw:
        continue

    if any(col.endswith(suffix) for suffix in ["_chg", "_z", "_lag1"]):
        features.append(col)
    elif not any(keyword in col for keyword in forbidden_keywords):
        features.append(col)

print(
    f"\nStrict feature selection kept {len(features)} features "
    f"(from {len(merged.columns) - 2} non-Date/non-target cols)."
)

# ----------------------------- Correlation-Based Pruning -----------------------------
def prune_high_corr_features(X_train: pd.DataFrame, y_train: pd.Series, threshold=0.9):
    corrs = {}
    for col in X_train.columns:
        x = X_train[col].values
        if np.all(np.isnan(x)) or np.nanstd(x) == 0:
            corrs[col] = 0.0
            continue
        corr = np.corrcoef(x, y_train.values)[0, 1]
        corrs[col] = corr

    corr_series = pd.Series(corrs)
    flagged = corr_series[np.abs(corr_series) > 0.8].sort_values(
        key=np.abs, ascending=False
    )

    drop_features = []
    keep_features = []
    for col, corr in flagged.items():
        if abs(corr) > threshold:
            drop_features.append(col)
        else:
            if col.endswith("_lag1"):
                keep_features.append(col)
            else:
                drop_features.append(col)

    print("\n⚠️ Correlation Audit (TRAIN ONLY):")
    print("Dropped features (dominance/leakage):")
    print(drop_features)
    if keep_features:
        print("\n✅ Kept high-correlation lagged macro/credit features:")
        print(keep_features)

    return drop_features


# IMPORTANT: compute correlations using TRAIN period only (avoid test leakage)
train_start = pd.Timestamp("2015-01-01")
train_end = pd.Timestamp("2021-12-31")
train_mask = (merged["Date"] >= train_start) & (merged["Date"] <= train_end)

X_train_audit = merged.loc[train_mask, features].astype(float)
y_train_audit = merged.loc[train_mask, "SPY_NextDay_Return"].astype(float)

cols_to_drop = prune_high_corr_features(X_train_audit, y_train_audit, threshold=0.9)

final_features = [c for c in features if c not in cols_to_drop]
print(f"\nFinal features after pruning: {len(final_features)}")

# Save feature list for reproducibility
pd.Series(final_features).to_csv("final_feature_list.csv", index=False, header=False)
print("Saved final_feature_list.csv")

# ------------------------------------------------------------------
# 9. FINAL STATS
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL CLEAN DATASET")
print("=" * 60)
print(f"Rows: {len(merged):,}")
print(f"Cols: {len(merged.columns):,}")
print(f"Date range: {merged['Date'].min().date()} → {merged['Date'].max().date()}")

final_miss = merged.drop(columns=["Date", "SPY_NextDay_Return"]).isnull().mean() * 100
print("\nTop 10 missing % (after cleaning):")
print(final_miss.round(2).sort_values(ascending=False).head(10))

# ------------------------------------------------------------------
# 10. SAVE
# ------------------------------------------------------------------
parquet_file = "master_spy_clean_final.parquet"
csv_file = "master_spy_clean_final.csv"

merged.to_parquet(parquet_file, index=False)
merged.to_csv(csv_file, index=False)

print("\nSAVED:")
print(f"  → {parquet_file} (Parquet, fast & small)")
print(f"  → {csv_file} (CSV, readable)")
