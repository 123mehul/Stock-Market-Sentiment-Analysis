# Simple Stock Sentiment Analyzer

A Python program that scrapes financial news headlines, analyzes their sentiment, and correlates it with stock price movements.

## Features

- 📰 Scrapes financial news headlines from Yahoo Finance
- 🧠 Analyzes sentiment using TextBlob NLP library
- 📈 Fetches real stock data using yfinance
- 🔗 Calculates correlation between sentiment and price movements
- 📊 Provides clear, formatted output with analysis results



## Usage

The program will:
1. Ask you to choose between default stocks or enter a custom symbol
2. Scrape recent news headlines for the selected stock(s)
3. Analyze sentiment of each headline (-1 to +1 scale)
4. Fetch recent stock price data
5. Calculate correlation between sentiment and price movements
6. Display comprehensive results

## Example Output

```
📊 SENTIMENT ANALYSIS RESULTS FOR AAPL
--------------------------------------------------
📰 Headlines Analyzed: 15
😊 Positive: 8 (53.3%)
😐 Neutral: 4 (26.7%)
😞 Negative: 3 (20.0%)
📈 Average Sentiment Score: 0.245

💰 STOCK PERFORMANCE
------------------------------
Current Price: $182.52
Daily Change: $2.15 (+1.19%)

🔗 SENTIMENT-PRICE CORRELATION
-----------------------------------
Correlation Coefficient: 0.342
Average Sentiment: 0.245
Average Return: 1.19%
Correlation Strength: Weak

📋 SAMPLE HEADLINES & SENTIMENT
----------------------------------------
😊 [+0.45] AAPL reports strong quarterly earnings beat...
😐 [+0.12] AAPL stock shows resilience despite economic headwinds...
😞 [-0.23] AAPL faces regulatory challenges ahead...
```

## How It Works

1. **News Scraping**: Uses BeautifulSoup to extract headlines from Yahoo Finance
2. **Sentiment Analysis**: TextBlob analyzes each headline's emotional tone
3. **Stock Data**: yfinance provides real-time and historical stock prices
4. **Correlation**: Calculates Pearson correlation between sentiment scores and returns
5. **Results**: Displays formatted analysis with interpretation


## Notes

- If web scraping fails, the program uses mock headlines for demonstration
- Requires internet connection for both news scraping and stock data
- Sentiment analysis is basic but effective for general market sentiment
- Correlation calculations use recent data to match sentiment timeframe
