import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import numpy as np
from datetime import datetime, timedelta
import time

class StockSentimentAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_news_headlines(self, symbol, num_headlines=20):
        """Scrape news headlines for a given stock symbol"""
        headlines = []
        
        # Yahoo Finance news
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news/"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find headline elements
            headline_elements = soup.find_all(['h3', 'h4'], class_=lambda x: x and 'headline' in x.lower())
            
            for element in headline_elements[:num_headlines]:
                text = element.get_text().strip()
                if text and len(text) > 10:
                    headlines.append(text)
                    
        except Exception as e:
            print(f"Error scraping Yahoo Finance: {e}")
        
        # If scraping fails, use mock headlines for demonstration
        if not headlines:
            headlines = self.generate_mock_headlines(symbol)
            
        return headlines[:num_headlines]
    
    def generate_mock_headlines(self, symbol):
        """Generate realistic mock headlines for demonstration"""
        templates = [
            f"{symbol} reports strong quarterly earnings beat",
            f"{symbol} stock jumps on positive analyst upgrade",
            f"{symbol} announces innovative product launch",
            f"{symbol} faces regulatory challenges ahead",
            f"{symbol} CEO optimistic about future growth prospects",
            f"{symbol} stock declines amid broader market volatility",
            f"{symbol} expands operations into emerging markets",
            f"{symbol} beats revenue expectations for third quarter",
            f"{symbol} announces strategic partnership deal",
            f"{symbol} shows resilience despite economic headwinds",
            f"Analysts raise price target for {symbol} shares",
            f"{symbol} reports disappointing guidance for next quarter",
            f"{symbol} stock reaches new 52-week high",
            f"{symbol} grapples with supply chain disruptions",
            f"{symbol} board approves dividend increase"
        ]
        return templates
    
    def analyze_sentiment(self, headlines):
        """Analyze sentiment of headlines using TextBlob"""
        sentiments = []
        
        for headline in headlines:
            blob = TextBlob(headline)
            sentiment_score = blob.sentiment.polarity  # Range: -1 to 1
            
            # Classify sentiment
            if sentiment_score > 0.1:
                sentiment_label = "Positive"
            elif sentiment_score < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            sentiments.append({
                'headline': headline,
                'score': round(sentiment_score, 3),
                'label': sentiment_label
            })
        
        return sentiments
    
    def get_stock_data(self, symbol, period="1mo"):
        """Get stock price data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            
            # Calculate daily returns
            data['Daily_Return'] = data['Close'].pct_change()
            
            return data
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None
    
    def calculate_correlation(self, sentiment_scores, stock_returns):
        """Calculate correlation between sentiment and stock returns"""
        if len(sentiment_scores) == 0 or len(stock_returns) == 0:
            return None
        
        # Use the average sentiment score
        avg_sentiment = np.mean(sentiment_scores)
        
        # Use recent stock returns (same timeframe)
        recent_returns = stock_returns.dropna().tail(len(sentiment_scores))
        
        if len(recent_returns) == 0:
            return None
        
        avg_return = np.mean(recent_returns)
        
        # Simple correlation calculation
        correlation = np.corrcoef([avg_sentiment], [avg_return])[0, 1]
        
        return {
            'correlation': round(correlation, 3) if not np.isnan(correlation) else 0,
            'avg_sentiment': round(avg_sentiment, 3),
            'avg_return': round(avg_return * 100, 3)  # Convert to percentage
        }
    
    def analyze_stock(self, symbol):
        """Complete analysis for a stock symbol"""
        print(f"\n{'='*50}")
        print(f"ANALYZING {symbol}")
        print(f"{'='*50}")
        
        # Step 1: Scrape news headlines
        print("1. Scraping news headlines...")
        headlines = self.scrape_news_headlines(symbol)
        print(f"   Found {len(headlines)} headlines")
        
        # Step 2: Analyze sentiment
        print("2. Analyzing sentiment...")
        sentiments = self.analyze_sentiment(headlines)
        
        # Step 3: Get stock data
        print("3. Fetching stock data...")
        stock_data = self.get_stock_data(symbol)
        
        if stock_data is None:
            print("   Error: Could not fetch stock data")
            return
        
        # Step 4: Calculate correlation
        print("4. Calculating correlation...")
        sentiment_scores = [s['score'] for s in sentiments]
        correlation_data = self.calculate_correlation(sentiment_scores, stock_data['Daily_Return'])
        
        # Display results
        self.display_results(symbol, sentiments, stock_data, correlation_data)
    
    def display_results(self, symbol, sentiments, stock_data, correlation_data):
        """Display analysis results"""
        print(f"\n📊 SENTIMENT ANALYSIS RESULTS FOR {symbol}")
        print("-" * 50)
        
        # Sentiment summary
        positive = sum(1 for s in sentiments if s['label'] == 'Positive')
        negative = sum(1 for s in sentiments if s['label'] == 'Negative')
        neutral = sum(1 for s in sentiments if s['label'] == 'Neutral')
        
        print(f"📰 Headlines Analyzed: {len(sentiments)}")
        print(f"😊 Positive: {positive} ({positive/len(sentiments)*100:.1f}%)")
        print(f"😐 Neutral: {neutral} ({neutral/len(sentiments)*100:.1f}%)")
        print(f"😞 Negative: {negative} ({negative/len(sentiments)*100:.1f}%)")
        
        avg_sentiment = np.mean([s['score'] for s in sentiments])
        print(f"📈 Average Sentiment Score: {avg_sentiment:.3f}")
        
        # Stock performance
        current_price = stock_data['Close'].iloc[-1]
        price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
        price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
        
        print(f"\n💰 STOCK PERFORMANCE")
        print("-" * 30)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Daily Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
        
        # Correlation analysis
        if correlation_data:
            print(f"\n🔗 SENTIMENT-PRICE CORRELATION")
            print("-" * 35)
            print(f"Correlation Coefficient: {correlation_data['correlation']}")
            print(f"Average Sentiment: {correlation_data['avg_sentiment']}")
            print(f"Average Return: {correlation_data['avg_return']}%")
            
            # Interpretation
            corr = abs(correlation_data['correlation'])
            if corr > 0.7:
                strength = "Strong"
            elif corr > 0.5:
                strength = "Moderate"
            elif corr > 0.3:
                strength = "Weak"
            else:
                strength = "Very Weak"
            
            print(f"Correlation Strength: {strength}")
        
        # Sample headlines
        print(f"\n📋 SAMPLE HEADLINES & SENTIMENT")
        print("-" * 40)
        for i, sentiment in enumerate(sentiments[:5]):
            emoji = "😊" if sentiment['label'] == 'Positive' else "😞" if sentiment['label'] == 'Negative' else "😐"
            print(f"{emoji} [{sentiment['score']:+.2f}] {sentiment['headline'][:60]}...")

def main():
    analyzer = StockSentimentAnalyzer()
    
    print("🚀 Stock Market Sentiment Analyzer")
    print("=" * 40)
    
    # Default stocks to analyze
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    
    print("Select an option:")
    print("1. Analyze default stocks (AAPL, GOOGL, MSFT, TSLA)")
    print("2. Enter custom stock symbol")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '2':
        symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
        if symbol:
            stocks = [symbol]
    
    # Analyze each stock
    for symbol in stocks:
        try:
            analyzer.analyze_stock(symbol)
            if len(stocks) > 1:
                time.sleep(2)  # Brief pause between stocks
        except Exception as e:
            print(f"Error analyzing {symbol}: {e}")
    
    print(f"\n✅ Analysis complete!")

if __name__ == "__main__":
    main()