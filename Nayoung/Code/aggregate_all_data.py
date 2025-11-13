"""
Script to aggregate all team members' data into a single comprehensive dataset for SPY prediction.

This script merges:
- Michael's SPY core data (processed features: gaps, indicators, trend, liquidity, volatility)
- Brian's sector data
- Daniel's market indicators
- Nayoung's macro positioning data
- Tee's FX flows data

Output: Complete aggregated dataset saved to Nayoung/Data/spy_complete_dataset.csv
"""

import pandas as pd
import os
from pathlib import Path

# Define base directory
BASE_DIR = Path(__file__).parent.parent.parent

def load_and_prepare_data():
    """Load all CSV files and prepare them for merging."""

    print("Loading data files...")

    # 1. Load Nayoung's macro positioning data (use as base - has AdjClose)
    print("  - Loading Nayoung's macro positioning data...")
    nayoung_df = pd.read_csv(BASE_DIR / 'Nayoung' / 'Data' / 'SPY_with_macro_positioning.csv')
    nayoung_df['Date'] = pd.to_datetime(nayoung_df['Date'])
    # Rename to avoid confusion - AdjClose is our primary SPY price
    nayoung_df = nayoung_df.rename(columns={
        'AdjClose': 'SPY_AdjClose',
        'Close': 'SPY_Close',
        'High': 'SPY_High',
        'Low': 'SPY_Low',
        'Open': 'SPY_Open',
        'Volume': 'SPY_Volume'
    })

    # 2. Load Michael's processed features
    print("  - Loading Michael's processed features...")

    # Gaps data
    gaps_df = pd.read_csv(BASE_DIR / 'Michael' / 'Data' / 'spy_rth_gaps_overnight_20141231-20250602.csv')
    gaps_df['Date'] = pd.to_datetime(gaps_df['Date'])
    # Drop duplicate OHLCV columns
    gaps_df = gaps_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore')

    # Indicators data
    indicators_df = pd.read_csv(BASE_DIR / 'Michael' / 'Data' / 'spy_rth_indicators_20141231-20250602.csv')
    indicators_df['Date'] = pd.to_datetime(indicators_df['Date'])
    # Drop duplicate OHLCV columns
    indicators_df = indicators_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore')

    # Trend and liquidity pressure data
    liquidity_df = pd.read_csv(BASE_DIR / 'Michael' / 'Data' / 'spy_rth_trend_liquidity_pressure_20141231-20250602.csv')
    liquidity_df['Date'] = pd.to_datetime(liquidity_df['Date'])
    # Drop duplicate OHLCV columns
    liquidity_df = liquidity_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore')

    # Trend and mean reversion data
    meanrev_df = pd.read_csv(BASE_DIR / 'Michael' / 'Data' / 'spy_rth_trend_meanrev_20141231-20250602.csv')
    meanrev_df['Date'] = pd.to_datetime(meanrev_df['Date'])
    # Drop duplicate OHLCV columns
    meanrev_df = meanrev_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore')

    # Volatility data
    volatility_df = pd.read_csv(BASE_DIR / 'Michael' / 'Data' / 'spy_rth_volatility_20141231-20250602.csv')
    volatility_df['Date'] = pd.to_datetime(volatility_df['Date'])
    # Drop duplicate OHLCV columns
    volatility_df = volatility_df.drop(columns=['Open', 'High', 'Low', 'Close', 'Volume'], errors='ignore')

    # 3. Load Brian's sector data
    print("  - Loading Brian's sector data...")
    brian_df = pd.read_csv(BASE_DIR / 'Brian' / 'Data' / 'spy_sector_data_Brian.csv')
    brian_df['Date'] = pd.to_datetime(brian_df['Date'])
    # Drop Prices_SPY as we're using AdjClose from Nayoung's data
    brian_df = brian_df.drop(columns=['Prices_SPY'], errors='ignore')

    # 4. Load Daniel's market indicators
    print("  - Loading Daniel's market indicators...")
    daniel_df = pd.read_csv(BASE_DIR / 'Daniel' / 'market_indicators_extended.csv')
    daniel_df['Date'] = pd.to_datetime(daniel_df['Date'])

    # 5. Load Tee's FX flows data
    print("  - Loading Tee's FX flows data...")
    tee_df = pd.read_csv(BASE_DIR / 'Tee' / 'Data' / 'spy_fx_flows.csv')
    # Convert time column to Date (extract date only)
    tee_df['Date'] = pd.to_datetime(tee_df['time']).dt.date
    tee_df['Date'] = pd.to_datetime(tee_df['Date'])
    # Drop duplicate OHLCV columns and original time column
    tee_df = tee_df.drop(columns=['time', 'close', 'high', 'low', 'open', 'volume'], errors='ignore')
    # Prefix FX columns to avoid confusion
    fx_cols = [col for col in tee_df.columns if col != 'Date']
    tee_df = tee_df.rename(columns={col: f'FX_{col}' for col in fx_cols})

    return nayoung_df, gaps_df, indicators_df, liquidity_df, meanrev_df, volatility_df, brian_df, daniel_df, tee_df


def merge_all_data(nayoung_df, gaps_df, indicators_df, liquidity_df, meanrev_df, volatility_df, brian_df, daniel_df, tee_df):
    """Merge all dataframes on Date column."""

    print("\nMerging all datasets...")

    # Start with Nayoung's data as base
    merged_df = nayoung_df.copy()
    print(f"  Base (Nayoung): {len(merged_df)} rows, {len(merged_df.columns)} columns")

    # Merge Michael's features
    print("  - Merging Michael's gaps data...")
    merged_df = merged_df.merge(gaps_df, on='Date', how='left', suffixes=('', '_gaps'))
    print(f"    After gaps: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    print("  - Merging Michael's indicators data...")
    merged_df = merged_df.merge(indicators_df, on='Date', how='left', suffixes=('', '_indicators'))
    print(f"    After indicators: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    print("  - Merging Michael's liquidity data...")
    merged_df = merged_df.merge(liquidity_df, on='Date', how='left', suffixes=('', '_liquidity'))
    print(f"    After liquidity: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    print("  - Merging Michael's mean reversion data...")
    merged_df = merged_df.merge(meanrev_df, on='Date', how='left', suffixes=('', '_meanrev'))
    print(f"    After mean reversion: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    print("  - Merging Michael's volatility data...")
    merged_df = merged_df.merge(volatility_df, on='Date', how='left', suffixes=('', '_volatility'))
    print(f"    After volatility: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    # Merge Brian's sector data
    print("  - Merging Brian's sector data...")
    merged_df = merged_df.merge(brian_df, on='Date', how='left', suffixes=('', '_brian'))
    print(f"    After Brian: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    # Merge Daniel's market indicators
    print("  - Merging Daniel's market indicators...")
    merged_df = merged_df.merge(daniel_df, on='Date', how='left', suffixes=('', '_daniel'))
    print(f"    After Daniel: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    # Merge Tee's FX flows data
    print("  - Merging Tee's FX flows data...")
    merged_df = merged_df.merge(tee_df, on='Date', how='left', suffixes=('', '_tee'))
    print(f"    After Tee: {len(merged_df)} rows, {len(merged_df.columns)} columns")

    # Handle any remaining duplicate columns
    duplicate_cols = [col for col in merged_df.columns if col.endswith(('_gaps', '_indicators', '_liquidity', '_meanrev', '_volatility', '_brian', '_daniel', '_tee'))]
    if duplicate_cols:
        print(f"\n  Warning: Found {len(duplicate_cols)} duplicate columns that will be removed:")
        print(f"    {duplicate_cols[:10]}{'...' if len(duplicate_cols) > 10 else ''}")
        merged_df = merged_df.drop(columns=duplicate_cols)

    # Sort by date
    merged_df = merged_df.sort_values('Date').reset_index(drop=True)

    return merged_df


def save_aggregated_data(merged_df):
    """Save the aggregated dataset."""

    output_path = BASE_DIR / 'Nayoung' / 'Data' / 'spy_complete_dataset.csv'

    print(f"\nSaving aggregated dataset to {output_path}...")
    merged_df.to_csv(output_path, index=False)

    print(f"[SUCCESS] Successfully saved!")
    print(f"\nFinal dataset summary:")
    print(f"  - Total rows: {len(merged_df)}")
    print(f"  - Total columns: {len(merged_df.columns)}")
    print(f"  - Date range: {merged_df['Date'].min()} to {merged_df['Date'].max()}")
    print(f"  - Missing values: {merged_df.isna().sum().sum()} ({merged_df.isna().sum().sum() / (len(merged_df) * len(merged_df.columns)) * 100:.2f}%)")

    # Show column groups
    print(f"\nColumn breakdown:")
    print(f"  - SPY OHLCV: SPY_AdjClose, SPY_Close, SPY_High, SPY_Low, SPY_Open, SPY_Volume")
    print(f"  - Macro indicators (Nayoung): CPI, CorePCE, Payrolls, Unemployment, etc.")
    print(f"  - Gap features (Michael): {len([c for c in merged_df.columns if 'gap' in c.lower()])} columns")
    print(f"  - Technical indicators (Michael): SMA, EMA, MACD, RSI, Stochastic, Bollinger Bands, etc.")
    print(f"  - Momentum features (Michael): {len([c for c in merged_df.columns if 'mom_' in c])} columns")
    print(f"  - Volatility features (Michael): {len([c for c in merged_df.columns if 'vol' in c.lower()])} columns")
    print(f"  - Sector data (Brian): {len([c for c in merged_df.columns if any(x in c for x in ['Prices_', 'Returns_', 'Ratios_', 'Features_', 'ZScores_'])])} columns")
    print(f"  - Market indicators (Daniel): VIX, VVIX, SKEW, UST yields, spreads, etc.")
    print(f"  - FX flows (Tee): {len([c for c in merged_df.columns if c.startswith('FX_')])} columns")

    return output_path


def main():
    """Main execution function."""

    print("="*80)
    print("SPY Complete Dataset Aggregation")
    print("="*80)

    # Load data
    nayoung_df, gaps_df, indicators_df, liquidity_df, meanrev_df, volatility_df, brian_df, daniel_df, tee_df = load_and_prepare_data()

    # Merge data
    merged_df = merge_all_data(nayoung_df, gaps_df, indicators_df, liquidity_df, meanrev_df, volatility_df, brian_df, daniel_df, tee_df)

    # Save data
    output_path = save_aggregated_data(merged_df)

    print("\n" + "="*80)
    print(f"Aggregation complete! Output saved to:")
    print(f"  {output_path}")
    print("="*80)


if __name__ == "__main__":
    main()
