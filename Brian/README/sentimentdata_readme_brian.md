# Tiingo News Sentiment Analysis with DistilRoBERTa

This project integrates **QuantConnect’s Tiingo News data** with a **pretrained DistilRoBERTa financial sentiment model** to generate daily sentiment scores for SPY-related news.  
The output is a dataset of news articles (limited to 10 per day) with sentiment scores, plus a daily average sentiment index.

---

## ⚙️ How the Code Works

1. **QuantBook Setup**
   - Creates a `QuantBook` instance.
   - Adds **SPY equity** and **Tiingo News** as data sources.

2. **Data Retrieval**
   - Defines a time window (`2015-01-01` to `2025-08-30`).
   - Pulls historical Tiingo news articles linked to SPY.
   - Ensures timestamps are properly parsed as UTC datetimes.

3. **Feature Engineering**
   - Combines `title` and `description` into a single `text` field.
   - Limits to **10 articles per day** to avoid overweighting high-volume days.
   - Keeps only relevant columns: `date`, `time`, `title`, `description`, `text`.

4. **Sentiment Model**
   - Loads **DistilRoBERTa fine-tuned on financial news sentiment**:
     - Model: `mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis`
   - Defines a scoring function:
     - Negative = -1  
     - Neutral = 0  
     - Positive = +1  
   - Applies the model to each article’s text.

5. **Daily Aggregation**
   - Computes the **average sentiment score per day** across all articles.

6. **Output**
   - Prints the full DataFrame of up to 10 articles per day with sentiment scores.
   - Provides a daily sentiment index (`daily_avg`).

---

## 📂 Example Output Structure

The CSV contains the following columns:

| Column           | Type      | Description                                                                 |
|------------------|-----------|-----------------------------------------------------------------------------|
| `date`           | datetime  | Trading day (UTC normalized).                                               |
| `time`           | datetime  | Timestamp of the news article (UTC).                                        |
| `sentiment_score`| float     | Continuous sentiment score between **-1 (negative)** and **+1 (positive)**. |

### Example rows

| date       | time                | sentiment_score |
|------------|---------------------|-----------------|
| 2015-01-01 | 00:00:00+00:00      | -0.330553       |
| 2015-01-02 | 00:00:00+00:00      | -0.100757       |
| 2015-01-03 | 00:00:00+00:00      | -0.109538       |
| 2015-01-04 | 00:00:00+00:00      | -0.398740       |
| 2015-01-05 | 00:00:00+00:00      | -0.171743       |

---

## ⚠️ Notes and Reminders

- **NaN / Empty Texts**: Articles with missing titles or descriptions are handled by filling with empty strings. If no text is available, sentiment defaults to `0` (neutral).
- **Daily Cap**: Only the first 10 articles per day are kept to avoid bias from high-volume news days.
- **Model Limitations**:
  - The sentiment model is trained on financial news but may misclassify nuanced or ambiguous headlines.
  - Scores are **continuous** between -1 and +1, but interpretation should be contextual.
- **Performance**: Running inference on large datasets may be slow without GPU acceleration.

---

## 📊 Use Cases

- **Market Sentiment Index**: Use `daily_avg` as a proxy for investor mood around SPY.
- **Event Studies**: Align sentiment spikes with SPY returns or volatility.
- **Feature Engineering**: Incorporate sentiment scores into trading models alongside price/volume features.

---


## 📊 Visualizing Daily Sentiment Trends

You can use the dataset to plot how sentiment evolves over time.  
The example below shows how to load the CSV and create a simple time series chart.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("daily_sentiment.csv", parse_dates=["date"])

# Plot daily sentiment trend
plt.figure(figsize=(12,6))
plt.plot(df["date"], df["sentiment_score"], label="Daily Avg Sentiment", color="blue")
plt.axhline(0, color="red", linestyle="--", label="Neutral")
plt.title("SPY News Sentiment Over Time")
plt.xlabel("Date")
plt.ylabel("Sentiment Score (-1 to +1)")
plt.legend()
plt.show()

---