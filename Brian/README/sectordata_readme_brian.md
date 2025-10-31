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


## CSV Format

The CSV is a **multi-level column dataset** with the following groups and columns:

---

### 🟦 Prices
Daily adjusted close prices for SPY and sector ETFs.

| Column         | Description                                      |
|----------------|--------------------------------------------------|
| `Prices_SPY`   | SPDR S&P 500 ETF (broad market benchmark).       |
| `Prices_XLK`   | Technology Select Sector ETF.                    |
| `Prices_XLF`   | Financials Select Sector ETF.                    |
| `Prices_XLE`   | Energy Select Sector ETF.                        |
| `Prices_XLV`   | Health Care Select Sector ETF.                   |
| `Prices_XLU`   | Utilities Select Sector ETF.                     |
| `Prices_XLY`   | Consumer Discretionary Select Sector ETF.        |
| `Prices_XLP`   | Consumer Staples Select Sector ETF.              |
| `Prices_XLB`   | Materials Select Sector ETF.                     |
| `Prices_XLRE`  | Real Estate Select Sector ETF.                   |
| `Prices_XLC`   | Communication Services Select Sector ETF.        |

---

### 🟩 Returns
Daily percentage returns for SPY and sector ETFs.

| Column          | Description                                      |
|-----------------|--------------------------------------------------|
| `Returns_SPY`   | Daily % return of SPY.                           |
| `Returns_XLK`   | Daily % return of XLK.                           |
| `Returns_XLF`   | Daily % return of XLF.                           |
| `Returns_XLE`   | Daily % return of XLE.                           |
| `Returns_XLV`   | Daily % return of XLV.                           |
| `Returns_XLU`   | Daily % return of XLU.                           |
| `Returns_XLY`   | Daily % return of XLY.                           |
| `Returns_XLP`   | Daily % return of XLP.                           |
| `Returns_XLB`   | Daily % return of XLB.                           |
| `Returns_XLRE`  | Daily % return of XLRE.                          |
| `Returns_XLC`   | Daily % return of XLC.                           |

---

### 🟨 Ratios
Relative strength ratios of sectors vs SPY.

| Column        | Description                                      |
|---------------|--------------------------------------------------|
| `Ratios_XLK_SPY` | XLK / SPY ratio (tech vs market).             |
| `Ratios_XLU_SPY` | XLU / SPY ratio (utilities vs market).        |
| `Ratios_XLF_SPY` | XLF / SPY ratio (financials vs market).       |

---

### 🟥 Features
Engineered macro/market features.

| Column                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `Features_SPY_Return`    | Daily % return of SPY (redundant for convenience).                          |
| `Features_GrowthMinusDef` | Spread between growth/cyclical sectors (XLK, XLF, XLE, XLY) and defensive sectors (XLU, XLP, XLV). |
| `Features_XLK_SPY`       | XLK / SPY ratio (duplicate from Ratios for modeling).                       |
| `Features_XLU_SPY`       | XLU / SPY ratio (duplicate from Ratios).                                    |
| `Features_XLF_SPY`       | XLF / SPY ratio (duplicate from Ratios).                                    |
| `Features_Breadth`       | Number of sectors outperforming SPY over the past 60 days.                  |
| `Features_Vol_Ratio_XLK_XLU` | Ratio of 20-day rolling volatility of XLK vs XLU.                      |
| `Features_DGS10`         | 10-year Treasury yield proxy (`^TNX` / 10).                                 |
| `Features_DGS5`          | 5-year Treasury yield proxy (`^FVX` / 10).                                  |
| `Features_Spread`        | Yield curve spread (10Y – 5Y).                                              |
| `Features_DefCyc_Ratio`  | Rolling ratio of defensive (XLP, XLU, XLV) vs cyclical (XLK, XLF, XLY).     |
| `Features_Avg_Sector_Corr` | 60-day rolling average of cross-sector correlations.                      |
| `Features_Crude_60dRet`  | 60-day return of crude oil futures (`CL=F`).                                |
| `Features_Copper_60dRet` | 60-day return of copper futures (`HG=F`).                                   |

---

### 🟪 ZScores
Standardized (z-scored) versions of the above features, using a **252-day rolling mean and standard deviation**.

| Column                   | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| `ZScores_SPY_Return`      | Z-score of SPY daily return.                                                |
| `ZScores_GrowthMinusDef`  | Z-score of Growth–Defensive spread.                                         |
| `ZScores_XLK_SPY`         | Z-score of XLK/SPY ratio.                                                   |
| `ZScores_XLU_SPY`         | Z-score of XLU/SPY ratio.                                                   |
| `ZScores_XLF_SPY`         | Z-score of XLF/SPY ratio.                                                   |
| `ZScores_Breadth`         | Z-score of breadth measure.                                                 |
| `ZScores_Vol_Ratio_XLK_XLU` | Z-score of volatility ratio (XLK vs XLU).                                |
| `ZScores_DGS10`           | Z-score of 10Y Treasury yield proxy.                                        |
| `ZScores_DGS5`            | Z-score of 5Y Treasury yield proxy.                                         |
| `ZScores_Spread`          | Z-score of yield curve spread (10Y – 5Y).                                   |
| `ZScores_DefCyc_Ratio`    | Z-score of defensive vs cyclical ratio.                                     |
| `ZScores_Avg_Sector_Corr` | Z-score of average sector correlation.                                      |
| `ZScores_Crude_60dRet`    | Z-score of 60-day crude oil return.                                         |
| `ZScores_Copper_60dRet`   | Z-score of 60-day copper return.                                            |

---

### Example rows

| Date       | Prices_SPY | Prices_XLB | ... | Returns_SPY | Ratios_XLK_SPY | Features_SPY_Return | Features_Breadth | ZScores_SPY_Return | ... |
|------------|------------|------------|-----|-------------|----------------|----------------------|------------------|---------------------|-----|
| 2015-02-01 | 171.09     | 39.19      | ... | NaN         | 0.2119         | NaN                  | 0                | NaN                 | ... |
| 2015-05-01 | 168.00     | 38.19      | ... | NaN         | 0.2125         | NaN                  | 0                | NaN                 | ... |
| 2015-06-01 | 166.42     | 37.85      | ... | NaN         | 0.2120         | NaN                  | 0                | NaN                 | ... |

---

## How to Use the CSV

Here’s a simple example of how to visualize one of the engineered features:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("market_features_master.csv", parse_dates=["Date"], index_col="Date")

# Plot Defensive vs Cyclical ratio
plt.figure(figsize=(12,6))
plt.plot(df.index, df["Features_DefCyc_Ratio"], label="Defensive vs Cyclical Ratio", color="green")
plt.title("Defensive vs Cyclical Ratio Over Time")
plt.xlabel("Date")
plt.ylabel("Ratio")
plt.legend()
plt.show()


---

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

