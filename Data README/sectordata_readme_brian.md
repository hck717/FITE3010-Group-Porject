# Sector Features Engineering Pipeline

This project builds a **feature engineering pipeline** for U.S. equity market analysis using sector ETFs, Treasury yields, and commodity proxies.  
The output is a **master dataset (`market_features_master.csv`)** that combines raw prices, returns, engineered features, and z-scored versions for modeling or research.

---

## ⚙️ How the Code Works

1. **Data Collection**
   - Downloads daily price data for **SPY** (S&P 500 ETF) and **sector ETFs** (XLK, XLF, XLE, XLV, XLU, XLY, XLP, XLB, XLRE, XLC).
   - Downloads **Treasury yield proxies** (`^TNX` for 10y, `^FVX` for 5y).
   - Downloads **commodity futures** (Crude Oil `CL=F`, Copper `HG=F`).

2. **Preprocessing**
   - Flattens Yahoo Finance’s multi-index columns.
   - Uses **Adjusted Close** prices if available, otherwise falls back to Close.
   - Computes **daily returns**.

3. **Feature Engineering**
   - **Relative Strength Ratios**: XLK/SPY, XLU/SPY, XLF/SPY.
   - **Growth vs. Defensive Proxy**: Rolling mean of cyclical vs. defensive sector returns.
   - **Seasonality**: Average monthly SPY returns.
   - **Breadth**: Number of sectors outperforming SPY over 60 days.
   - **Volatility Ratio**: Tech vs. Utilities volatility.
   - **Treasury Yields**: 10y, 5y, and spread.
   - **Commodities**: 60-day returns of crude oil and copper.
   - **Defensive vs. Cyclical Ratio**: Rolling ratio of defensive vs. cyclical ETF baskets.
   - **Cross-Sector Correlation**: Average 60-day correlation across sectors.

4. **Z-Scoring**
   - Each feature is standardized using a **252-day rolling mean and standard deviation**.
   - Produces a z-scored version of each feature for comparability.

5. **Master Dataset**
   - Combines:
     - Raw prices
     - Returns
     - Ratios
     - Engineered features
     - Z-scores
   - Saves to **`market_features_master.csv`**.

---

## 📂 Using the Generated CSV

The file `market_features_master.csv` contains a **multi-level column structure**:

**Please be reminded ther are Null or NaN values in the dataset, please handle it if you need to 100% continous dataset.

- **Prices**: Adjusted close prices of SPY and sector ETFs.
- **Returns**: Daily % changes.
- **Ratios**: Relative strength ratios (e.g., XLK/SPY).
- **Features**: Engineered features (see table below).
- **ZScores**: Standardized versions of features.

### Example Usage in Python

```python
import pandas as pd

# Load dataset
df = pd.read_csv("market_features_master.csv", index_col=0, parse_dates=True)

# Access engineered features
features = df["Features"]

# Example: Plot Growth vs Defensive feature
features["GrowthMinusDef"].plot(title="Growth vs Defensive Rotation")




## 📊 Feature Intuition Table

This table explains the intuition behind each engineered feature, its meaning, and the directional relationship with SPY.

| Feature              | Description | Why It Matters for SPY | Directional Relationship (↑ Feature → Effect on SPY) |
|-----------------------|-------------|-------------------------|------------------------------------------------------|
| **SPY_Return**        | Daily % change in SPY | Baseline target variable for analysis | N/A (this is SPY) |
| **GrowthMinusDef**    | Growth/cyclical sectors (XLK, XLF, XLE, XLY) minus defensive (XLU, XLP, XLV) | Captures risk‑on vs. risk‑off rotation | ↑ GrowthMinusDef → SPY ↑ (risk‑on) |
| **XLK_SPY**           | Tech vs. SPY relative strength | Tech leadership often drives market rallies | ↑ XLK_SPY → SPY ↑ |
| **XLU_SPY**           | Utilities vs. SPY relative strength | Utilities outperform in defensive regimes | ↑ XLU_SPY → SPY ↓ |
| **XLF_SPY**           | Financials vs. SPY relative strength | Financial strength signals economic optimism | ↑ XLF_SPY → SPY ↑ |
| **Breadth**           | # of sectors outperforming SPY over 60 days | Broad participation = healthier rallies | ↑ Breadth → SPY ↑ |
| **Vol_Ratio_XLK_XLU** | Volatility of Tech vs. Utilities | Higher tech vol signals risk appetite or instability | ↑ Vol_Ratio → mixed: can precede SPY ↑ (risk‑on) or SPY ↓ (instability) |
| **DGS10**             | 10‑year Treasury yield | Rising long rates can pressure equities via discounting | ↑ DGS10 → SPY ↓ (generally) |
| **DGS5**              | 5‑year Treasury yield | Tracks Fed policy expectations | ↑ DGS5 → SPY ↓ (tighter policy) |
| **Spread (10y–5y)**   | Yield curve slope | Steepening = growth optimism; inversion = recession risk | ↑ Spread → SPY ↑ |
| **DefCyc_Ratio**      | Defensive vs. cyclical sector ratio | Defensive leadership signals caution | ↑ DefCyc_Ratio → SPY ↓ |
| **Avg_Sector_Corr**   | Average 60‑day correlation across sectors | High correlation = systemic risk; low = diversification | ↑ Corr → SPY ↓ (fragile rallies) |
| **Crude_60dRet**      | 60‑day return of crude oil | Oil strength can mean growth or inflation | ↑ Crude → SPY ↑ if growth, ↓ if inflation fears |
| **Copper_60dRet**     | 60‑day return of copper | “Dr. Copper” = global growth barometer | ↑ Copper → SPY ↑ |
| **Seasonality**       | Avg monthly SPY returns | Captures recurring calendar effects | ↑ Seasonality (positive months) → SPY ↑ |

