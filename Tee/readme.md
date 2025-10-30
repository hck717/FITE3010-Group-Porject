# 🧠 SPY Multi-Asset Feature Engineering Pipeline

This project builds a **feature engineering pipeline** for SPY (S&P 500 ETF) using cross-asset signals such as money flow, commodities, FX strength, and market breadth.  
The output file, **`spy_fx_flows.csv`**, contains a synchronized daily dataset of market-close values from 2015 to 2025 for model training and research.

---

## ⚙️ How the Code Works

### Data Collection (QuantConnect Research Environment)
All data are downloaded directly from QuantConnect’s institutional-grade feeds

**Assets subscribed**
- **SPY** – S&P 500 ETF (market-cap benchmark)  
- **RSP** – Equal-weight S&P 500 ETF (breadth proxy)  
- **USO** – Crude Oil ETF (commodity proxy)  
- **GLD** – Gold ETF (risk-off / inflation hedge)  
- **EURUSD**, **USDJPY** – Major FX pairs for USD strength  

Each symbol uses **daily resolution**, corporate-action-adjusted prices, and the **NYSE trading calendar**.  
Timestamps are in **UTC**, corresponding to U.S. market close (4 PM Eastern Time).

---

### 🧹 Preprocessing
- Converts QuantConnect’s multi-index output into a tidy `DataFrame` with columns:  
  `open, high, low, close, volume`.
- Removes timezone info and missing rows for non-trading days.
- Merges all data sources on **New York trading days** (`ny_date` alignment).
- Ensures every feature aligns perfectly across SPY, RSP, USO, GLD, and FX pairs.

---

### 🔬 Feature Engineering

| Category | Feature | Description |
|-----------|----------|-------------|
| **Price & Volume** | `open`, `high`, `low`, `close`, `volume` | Raw SPY OHLCV data (unscaled) |
| **Flow Volatility** | `flow_vol_z20` | 20-day rolling z-score of flow volatility; measures abnormal ETF activity |
| **Money Flow Proxy** | `flow_money_proxy5` | 5-day smoothed proxy for net capital inflow/outflow into SPY |
| **Commodities** | `uso_ret5`, `gld_ret5` | 5-day returns of crude oil and gold ETFs |
| **FX & USD** | `eurusd_ret5`, `usdjpy_ret5`, `usd_strength` | FX-based measures of USD strength; stronger USD often pressures risk assets |
| **Breadth** | `breadth_proxy` | Ratio of equal-weight / cap-weight S&P 500 (RSP ÷ SPY); higher = broader participation |

---

### 📏 Feature Scaling
Columns already standardized:  
`flow_vol_z20`, `flow_money_proxy5`, all `_ret5` columns, `usd_strength`, `breadth_proxy`.

Raw price columns (`open`, `high`, `low`, `close`, `volume`) should be standardized later using `StandardScaler` before model training.

---

### 💾 Output
- **File name:** `spy_features_full_v2.csv`  
- **Shape:** ~ 2,600 rows × 14 columns  
- **Each row:** one NYSE trading day  
- **Each column:** one engineered feature for SPY prediction

---

## 📂 Using the Generated CSV

```python
import pandas as pd

# Load dataset
df = pd.read_csv("spy_fx_flows.csv", parse_dates=["time"])
df.set_index("time", inplace=True)

# View data
print(df.head())

# Convert UTC → America/New_York for readability
df.index = df.index.tz_convert('America/New_York')







## 📊 Feature Intuition Table

This table explains the intuition behind each engineered feature, its meaning, and the directional relationship with SPY.

| Feature | Description | Why It Matters for SPY | Directional Relationship (↑ Feature → SPY) |
|----------|--------------|-------------------------|---------------------------------------------|
| **open / high / low / volume** | SPY OHLCV features | Captures daily market action and trading activity | ↑ Volume with ↑ Price = bullish confirmation |
| **close** | SPY closing price | Baseline market level | N/A |
| **flow_vol_z20** | 20-day z-score of ETF flow volatility | Detects abnormal fund inflows/outflows; high = crowded trade | Mixed – extremes can precede reversals |
| **flow_money_proxy5** | 5-day smoothed flow strength | Captures short-term liquidity entering or exiting equities | ↑ = SPY ↑ (bullish liquidity) |
| **uso_ret5** | 5-day crude-oil ETF return | Growth proxy; strong oil often signals economic expansion | ↑ = SPY ↑ (growth) or ↓ (inflation pressure) |
| **gld_ret5** | 5-day gold ETF return | Defensive / inflation-hedge asset | ↑ = SPY ↓ (risk-off move) |
| **eurusd_ret5** | 5-day EUR/USD return | Weaker USD (↑ EURUSD) improves U.S. exports and earnings | ↑ = SPY ↑ |
| **usdjpy_ret5** | 5-day USD/JPY return | Yen weakness fuels carry-trade risk-on flows | ↑ = SPY ↑ |
| **usd_strength** | Composite USD index (z-score) | Measures broad USD pressure on global liquidity | ↑ = SPY ↓ (tighter liquidity) |
| **breadth_proxy** | RSP/SPY − 1 ratio | Tracks equal-weight vs cap-weight leadership; breadth confirmation | ↑ = SPY ↑ (broad participation) |
