from AlgorithmImports import *
from QuantConnect.DataSource import TiingoNews
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Create QuantBook instance
qb = QuantBook()

# 2. Add SPY equity and Tiingo News
spy = qb.AddEquity("SPY", Resolution.Daily).Symbol
news = qb.AddData(TiingoNews, spy).Symbol

# 3. Define time window
start = pd.Timestamp(2015, 1, 1)
end   = pd.Timestamp(2025, 8, 30)

# 4. Pull history of news articles
news_history = qb.History(news, start, end).reset_index()

# 5. Ensure datetime type for 'time'
news_history["time"] = pd.to_datetime(news_history["time"], utc=True)

# 6. Feature engineering: combine title + description
news_history["text"] = news_history["title"].fillna("") + ". " + news_history["description"].fillna("")

# 7. Limit to max 10 news per day
news_history["date"] = news_history["time"].dt.normalize()
limited = (
    news_history
    .groupby("date", group_keys=False)
    .apply(lambda g: g.head(10))
    .reset_index(drop=True)
)

# 8. Keep only useful columns
features = limited[["date", "time", "title", "description", "text"]].copy()

# 9. Load DistilRoBERTa financial sentiment model
tokenizer = AutoTokenizer.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)
model = AutoModelForSequenceClassification.from_pretrained(
    "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
)
model.eval()

def score_with_distilroberta_continuous(text: str) -> float:
    if not text:
        return 0
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).flatten().tolist()
    return (-1 * probs[0]) + (0 * probs[1]) + (1 * probs[2])

# 10. Score each article
features["sentiment_score"] = features["text"].apply(score_with_distilroberta_continuous)

# 11. Compute daily average sentiment
daily_avg = features.groupby("date")["sentiment_score"].mean().reset_index()

# 12. Configure pandas to show all rows/columns
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
