# Dataset Documentation: SPY + Macro & Growth Features

This dataset combines historical price data for **SPY (S&P 500 ETF)** with multiple **U.S. macroeconomic indicators** and light technical features.  
It is used to build machine learning models that predict the *next-day price of SPY*.

---

## 📌 Columns Overview

| Column Name | Type | Description |
|-------------|-------|-------------|
| `Open` | float | SPY opening price of the day (regular trading hours). |
| `High` | float | Highest price SPY reached during the trading day. |
| `Low` | float | Lowest price SPY reached during the trading day. |
| `Close` | float | SPY closing price at market close (4 PM ET). |
| `AdjClose` | float | Adjusted close price accounting for dividends & splits. |
| `Volume` | int | Number of SPY shares traded that day. |

---

## 🏦 Macro & Growth Indicators (Fundamental / Economic Data)

| Column | Source | Meaning | Why Relevant to Stocks? |
|--------|---------|---------|--------------------------|
| `CPI` | BLS | Consumer Price Index (inflation rate of goods & services). | High CPI → higher inflation → Fed raises rates → stocks fall. |
| `CorePCE` | BEA | Core Personal Consumption Expenditure (inflation excluding food & energy). | Fed's *preferred* inflation measure; affects rate policy. |
| `Payrolls` | BLS | Non-farm payroll employment level (job creation). | Strong jobs → strong economy → stocks rise (unless Fed tightens). |
| `Unemployment` | BLS | U.S. unemployment rate % | Rising unemployment → recession risk → stocks drop. |
| `RetailSales` | Census | Total sales from U.S. retailers. | Reflects consumer demand (70% of GDP). |
| `DurGoodsOrders` | Census | Orders for long-lasting goods (cars, appliances, machinery). | Leading indicator of manufacturing strength. |
| `CoreCapexOrders` | Census | Business spending on capital goods (ex-aircraft & defense). | Proxy for corporate investment confidence. |
| `IndProd_Manufacturing` | Fed | US Industrial Production Index (manufacturing output). | Measures real economy activity & business cycle. |
| `Yield10` | U.S. Treasury | 10-year government bond yield (%) | Long-term borrowing cost, risk-free rate for valuation. |
| `Yield2` | U.S. Treasury | 2-year government bond yield (%) | Sensitive to Fed rate policy expectations. |
| `TermSpread` | computed | `Yield10 - Yield2` (also called *Yield Curve*) | Negative spread = recession signal (inverted curve). |
| `VIX` | CBOE | Volatility Index (expected future market fear). | High VIX → high uncertainty → stocks decline. |

---

## 📈 Engineered Technical Features

| Column | Meaning | How it's calculated |
|--------|---------|---------------------|
| `LogClose` | Natural log of SPY closing price | `log(Close)` |
| `Ret1` | 1-day log return | `LogClose[t] - LogClose[t-1]` |
| `Vol10` | 10-day rolling volatility (std of returns) | `std(Ret1 last 10 days)` |
| `y_next_log_close` | Machine learning target: **next-day log close price** | `LogClose.shift(-1)` |


## 📅 Data Frequency & Alignment

| Type | Frequency |
|-------|-----------|
| SPY price data | Daily (market trading days) |
| Macro series (CPI, Payrolls, etc.) | Mostly monthly → **shifted by 1 day** to prevent look-ahead bias |
| Bond yields, VIX | Daily |
| Engineered features | Daily (forward-filled where needed) |
