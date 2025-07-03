import requests
from bs4 import BeautifulSoup
import yfinance as yf
import pandas as pd
from textblob import TextBlob
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff

class EnhancedStockSentimentAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        # Power BI inspired color palette
        self.colors = {
            'primary': '#0078D4',
            'secondary': '#106EBE',
            'accent': '#00BCF2',
            'positive': '#107C10',
            'negative': '#D13438',
            'neutral': '#FF8C00',
            'background': '#F8F9FA',
            'text': '#323130',
            'grid': '#E1E1E1'
        }
        
        # Set professional styling
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette([self.colors['primary'], self.colors['accent'], self.colors['positive'], self.colors['negative']])
    
    def scrape_news_headlines(self, symbol, num_headlines=25):
        """Scrape news headlines for a given stock symbol"""
        headlines = []
        
        try:
            url = f"https://finance.yahoo.com/quote/{symbol}/news/"
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            headline_elements = soup.find_all(['h3', 'h4'], class_=lambda x: x and 'headline' in x.lower())
            
            for element in headline_elements[:num_headlines]:
                text = element.get_text().strip()
                if text and len(text) > 10:
                    headlines.append(text)
                    
        except Exception as e:
            print(f"Error scraping Yahoo Finance: {e}")
        
        if not headlines:
            headlines = self.generate_mock_headlines(symbol)
            
        return headlines[:num_headlines]
    
    def generate_mock_headlines(self, symbol):
        """Generate realistic mock headlines for demonstration"""
        templates = [
            f"{symbol} reports exceptional quarterly earnings, beats estimates by 15%",
            f"{symbol} stock surges on breakthrough product announcement",
            f"{symbol} announces revolutionary AI integration across platforms",
            f"{symbol} faces temporary regulatory review, stock dips slightly",
            f"{symbol} CEO delivers optimistic outlook at investor conference",
            f"{symbol} stock shows resilience amid broader market volatility",
            f"{symbol} expands global operations with $2B investment plan",
            f"{symbol} crushes revenue expectations for consecutive quarter",
            f"{symbol} forms strategic alliance with industry leader",
            f"{symbol} demonstrates strong fundamentals despite headwinds",
            f"Wall Street analysts upgrade {symbol} with raised price targets",
            f"{symbol} provides conservative guidance amid economic uncertainty",
            f"{symbol} reaches milestone 52-week high on strong performance",
            f"{symbol} successfully navigates supply chain challenges",
            f"{symbol} board approves substantial dividend increase",
            f"{symbol} launches comprehensive sustainability initiative",
            f"{symbol} stock experiences healthy consolidation after rally",
            f"{symbol} management outlines ambitious growth strategy",
            f"{symbol} receives overwhelmingly positive analyst coverage",
            f"{symbol} maintains competitive edge in challenging environment",
            f"{symbol} reports record-breaking user engagement metrics",
            f"{symbol} announces major acquisition to expand market share",
            f"{symbol} stock benefits from sector rotation into growth",
            f"{symbol} delivers impressive margin expansion results",
            f"{symbol} positioned for long-term success, analysts say"
        ]
        return templates
    
    def analyze_sentiment(self, headlines):
        """Analyze sentiment of headlines using TextBlob"""
        sentiments = []
        
        for i, headline in enumerate(headlines):
            blob = TextBlob(headline)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.15:
                sentiment_label = "Positive"
            elif sentiment_score < -0.15:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            sentiments.append({
                'headline': headline,
                'score': round(sentiment_score, 3),
                'label': sentiment_label,
                'index': i,
                'magnitude': abs(sentiment_score)
            })
        
        return sentiments
    
    def get_stock_data(self, symbol, period="2mo"):
        """Get stock price data using yfinance"""
        try:
            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            data['Daily_Return'] = data['Close'].pct_change()
            data['MA_7'] = data['Close'].rolling(window=7).mean()
            data['MA_21'] = data['Close'].rolling(window=21).mean()
            return data
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return None
    
    def create_executive_dashboard(self, sentiments, stock_data, symbol):
        """Create a comprehensive Power BI-style executive dashboard"""
        fig = make_subplots(
            rows=3, cols=3,
            subplot_titles=(
                f'{symbol} Stock Performance', 'Sentiment Distribution', 'Sentiment Timeline',
                'Price vs Moving Averages', 'Volume Analysis', 'Sentiment Intensity',
                'Daily Returns Distribution', 'Correlation Scatter', 'Key Metrics'
            ),
            specs=[
                [{"colspan": 2}, None, {"type": "pie"}],
                [{"secondary_y": True}, {"type": "bar"}, {"type": "scatter"}],
                [{"type": "histogram"}, {"type": "scatter"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # 1. Stock Performance (Candlestick)
        fig.add_trace(
            go.Candlestick(
                x=stock_data.index,
                open=stock_data['Open'],
                high=stock_data['High'],
                low=stock_data['Low'],
                close=stock_data['Close'],
                name='Price',
                increasing_line_color=self.colors['positive'],
                decreasing_line_color=self.colors['negative']
            ),
            row=1, col=1
        )
        
        # 2. Sentiment Distribution (Pie Chart)
        sentiment_counts = pd.Series([s['label'] for s in sentiments]).value_counts()
        fig.add_trace(
            go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.4,
                marker_colors=[self.colors['positive'], self.colors['neutral'], self.colors['negative']],
                textinfo='label+percent',
                textfont_size=10
            ),
            row=1, col=3
        )
        
        # 3. Price vs Moving Averages
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['Close'],
                mode='lines',
                name='Close Price',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['MA_7'],
                mode='lines',
                name='7-Day MA',
                line=dict(color=self.colors['accent'], width=1, dash='dash')
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=stock_data.index,
                y=stock_data['MA_21'],
                mode='lines',
                name='21-Day MA',
                line=dict(color=self.colors['secondary'], width=1, dash='dot')
            ),
            row=2, col=1
        )
        
        # 4. Volume Analysis
        fig.add_trace(
            go.Bar(
                x=stock_data.index,
                y=stock_data['Volume'],
                name='Volume',
                marker_color=self.colors['accent'],
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # 5. Sentiment Intensity Scatter
        sentiment_scores = [s['score'] for s in sentiments]
        magnitudes = [s['magnitude'] for s in sentiments]
        fig.add_trace(
            go.Scatter(
                x=sentiment_scores,
                y=magnitudes,
                mode='markers',
                marker=dict(
                    size=10,
                    color=sentiment_scores,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Sentiment", x=1.02)
                ),
                name='Intensity'
            ),
            row=2, col=3
        )
        
        # 6. Daily Returns Distribution
        returns = stock_data['Daily_Return'].dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=20,
                name='Returns',
                marker_color=self.colors['primary'],
                opacity=0.7
            ),
            row=3, col=1
        )
        
        # 7. Correlation Scatter (Sentiment vs Returns)
        daily_sentiments = []
        for i in range(len(returns)):
            if i < len(sentiment_scores):
                daily_sentiments.append(sentiment_scores[i])
            else:
                daily_sentiments.append(np.mean(sentiment_scores))
        
        daily_sentiments = daily_sentiments[:len(returns)]
        
        fig.add_trace(
            go.Scatter(
                x=daily_sentiments,
                y=returns,
                mode='markers',
                marker=dict(
                    size=8,
                    color=returns,
                    colorscale='RdYlGn',
                    showscale=False
                ),
                name='Sentiment vs Returns'
            ),
            row=3, col=2
        )
        
        # 8. Key Metrics Indicator
        current_price = stock_data['Close'].iloc[-1]
        price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
        avg_sentiment = np.mean(sentiment_scores)
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=current_price,
                delta={'reference': current_price - price_change, 'relative': True},
                title={"text": f"{symbol} Price"},
                number={'prefix': "$", 'font': {'size': 24}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=3, col=3
        )
        
        # Update layout with Power BI styling
        fig.update_layout(
            title={
                'text': f'{symbol} Executive Sentiment & Performance Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 24, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10, color=self.colors['text']),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=1000,
            margin=dict(t=100, b=50, l=50, r=50)
        )
        
        # Update axes styling
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=self.colors['grid'])
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=self.colors['grid'])
        
        fig.show()
    
    def create_sentiment_timeline_chart(self, sentiments, symbol):
        """Create a dedicated sentiment timeline chart"""
        sentiment_scores = [s['score'] for s in sentiments]
        colors = [self.colors['positive'] if s > 0.15 else self.colors['negative'] if s < -0.15 else self.colors['neutral'] for s in sentiment_scores]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(sentiment_scores))),
            y=sentiment_scores,
            mode='markers+lines',
            marker=dict(
                size=10,
                color=colors,
                line=dict(width=2, color='white')
            ),
            line=dict(width=2, color=self.colors['primary']),
            name='Sentiment Score'
        ))
        
        # Add horizontal lines for sentiment thresholds
        fig.add_hline(y=0.15, line_dash="dash", line_color=self.colors['positive'], 
                     annotation_text="Positive Threshold")
        fig.add_hline(y=-0.15, line_dash="dash", line_color=self.colors['negative'], 
                     annotation_text="Negative Threshold")
        fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                     annotation_text="Neutral")
        
        fig.update_layout(
            title=f'{symbol} Sentiment Score Timeline',
            xaxis_title='News Article Index',
            yaxis_title='Sentiment Score',
            yaxis=dict(range=[-1, 1]),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", color=self.colors['text'])
        )
        
        fig.show()
    
    def create_sentiment_kpi_dashboard(self, sentiments, symbol):
        """Create a KPI-focused dashboard like Power BI"""
        fig = make_subplots(
            rows=2, cols=4,
            subplot_titles=(
                'Sentiment Score', 'Positive %', 'Negative %', 'Neutral %',
                'Sentiment Trend', 'Score Distribution', 'Intensity Heatmap', 'Summary Stats'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                [{"colspan": 2}, None, {"type": "scatter"}, {"type": "table"}]
            ],
            vertical_spacing=0.15
        )
        
        # Calculate KPIs
        scores = [s['score'] for s in sentiments]
        labels = [s['label'] for s in sentiments]
        
        avg_score = np.mean(scores)
        positive_pct = (labels.count('Positive') / len(labels)) * 100
        negative_pct = (labels.count('Negative') / len(labels)) * 100
        neutral_pct = (labels.count('Neutral') / len(labels)) * 100
        
        # KPI Indicators
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=avg_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Sentiment"},
                gauge={
                    'axis': {'range': [-1, 1]},
                    'bar': {'color': self.colors['primary']},
                    'steps': [
                        {'range': [-1, -0.3], 'color': self.colors['negative']},
                        {'range': [-0.3, 0.3], 'color': self.colors['neutral']},
                        {'range': [0.3, 1], 'color': self.colors['positive']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0
                    }
                }
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=positive_pct,
                title={"text": "Positive %"},
                number={'suffix': "%", 'font': {'size': 20, 'color': self.colors['positive']}},
                delta={'reference': 33.33, 'relative': False}
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=negative_pct,
                title={"text": "Negative %"},
                number={'suffix': "%", 'font': {'size': 20, 'color': self.colors['negative']}},
                delta={'reference': 33.33, 'relative': False}
            ),
            row=1, col=3
        )
        
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=neutral_pct,
                title={"text": "Neutral %"},
                number={'suffix': "%", 'font': {'size': 20, 'color': self.colors['neutral']}},
                delta={'reference': 33.33, 'relative': False}
            ),
            row=1, col=4
        )
        
        # Sentiment Trend
        fig.add_trace(
            go.Scatter(
                x=list(range(len(scores))),
                y=scores,
                mode='lines+markers',
                line=dict(color=self.colors['primary'], width=3),
                marker=dict(size=6, color=self.colors['accent']),
                name='Sentiment Trend',
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Score Distribution
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=15,
                marker_color=self.colors['primary'],
                opacity=0.8,
                name='Distribution'
            ),
            row=2, col=2
        )
        
        # Intensity Heatmap (as scatter plot)
        magnitudes = [s['magnitude'] for s in sentiments]
        fig.add_trace(
            go.Scatter(
                x=scores,
                y=magnitudes,
                mode='markers',
                marker=dict(
                    size=8,
                    color=scores,
                    colorscale='RdYlGn',
                    showscale=True
                ),
                name='Intensity'
            ),
            row=2, col=3
        )
        
        # Summary Table
        summary_data = [
            ['Metric', 'Value'],
            ['Total Headlines', len(sentiments)],
            ['Avg Score', f"{avg_score:.3f}"],
            ['Max Score', f"{max(scores):.3f}"],
            ['Min Score', f"{min(scores):.3f}"],
            ['Std Dev', f"{np.std(scores):.3f}"],
            ['Positive Count', labels.count('Positive')],
            ['Negative Count', labels.count('Negative')],
            ['Neutral Count', labels.count('Neutral')]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value'],
                           fill_color=self.colors['primary'],
                           font=dict(color='white', size=12)),
                cells=dict(values=list(zip(*summary_data[1:])),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=11))
            ),
            row=2, col=4
        )
        
        fig.update_layout(
            title={
                'text': f'{symbol} Sentiment Analytics Dashboard',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 22, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10),
            height=800,
            showlegend=False
        )
        
        fig.show()
    
    def create_correlation_analysis(self, sentiments, stock_data, symbol):
        """Create detailed correlation analysis visualization"""
        sentiment_scores = [s['score'] for s in sentiments]
        returns = stock_data['Daily_Return'].dropna() * 100
        
        # Create synthetic daily sentiment data for correlation
        daily_sentiments = []
        for i in range(len(returns)):
            if i < len(sentiment_scores):
                daily_sentiments.append(sentiment_scores[i])
            else:
                daily_sentiments.append(np.mean(sentiment_scores))
        
        daily_sentiments = daily_sentiments[:len(returns)]
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Sentiment vs Returns Correlation',
                'Rolling Correlation (7-day)',
                'Sentiment & Price Overlay',
                'Statistical Summary'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"secondary_y": True}, {"type": "table"}]
            ]
        )
        
        # Correlation scatter plot
        correlation = np.corrcoef(daily_sentiments, returns)[0, 1]
        
        fig.add_trace(
            go.Scatter(
                x=daily_sentiments,
                y=returns,
                mode='markers',
                marker=dict(
                    size=8,
                    color=returns,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Returns %")
                ),
                name=f'Correlation: {correlation:.3f}'
            ),
            row=1, col=1
        )
        
        # Add trend line
        z = np.polyfit(daily_sentiments, returns, 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=daily_sentiments,
                y=p(daily_sentiments),
                mode='lines',
                name='Trend Line',
                line=dict(color=self.colors['negative'], dash='dash', width=2)
            ),
            row=1, col=1
        )
        
        # Rolling correlation
        df = pd.DataFrame({'sentiment': daily_sentiments, 'returns': returns})
        rolling_corr = df['sentiment'].rolling(window=7).corr(df['returns'])
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rolling_corr))),
                y=rolling_corr,
                mode='lines',
                name='7-day Rolling Correlation',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=2
        )
        
        # Sentiment & Price overlay
        normalized_sentiment = np.array(daily_sentiments) * 100  # Scale for visibility
        normalized_price = (stock_data['Close'].iloc[-len(returns):] / stock_data['Close'].iloc[-len(returns)]) * 100
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(normalized_sentiment))),
                y=normalized_sentiment,
                mode='lines',
                name='Sentiment (scaled)',
                line=dict(color=self.colors['accent'], width=2)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(normalized_price))),
                y=normalized_price,
                mode='lines',
                name='Price (normalized)',
                line=dict(color=self.colors['primary'], width=2),
                yaxis='y2'
            ),
            row=2, col=1
        )
        
        # Statistical summary table
        stats_data = [
            ['Statistic', 'Value'],
            ['Correlation Coefficient', f"{correlation:.4f}"],
            ['R-squared', f"{correlation**2:.4f}"],
            ['Avg Sentiment', f"{np.mean(daily_sentiments):.3f}"],
            ['Avg Return', f"{np.mean(returns):.2f}%"],
            ['Sentiment Volatility', f"{np.std(daily_sentiments):.3f}"],
            ['Return Volatility', f"{np.std(returns):.2f}%"],
            ['Max Sentiment', f"{max(daily_sentiments):.3f}"],
            ['Min Sentiment', f"{min(daily_sentiments):.3f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value'],
                           fill_color=self.colors['primary'],
                           font=dict(color='white', size=12)),
                cells=dict(values=list(zip(*stats_data[1:])),
                          fill_color='white',
                          font=dict(color=self.colors['text'], size=11))
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': f'{symbol} Advanced Correlation Analysis',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'color': self.colors['text']}
            },
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family="Segoe UI, Arial", size=10),
            height=800
        )
        
        fig.update_yaxes(title_text="Returns (%)", row=2, col=1)
        fig.update_yaxes(title_text="Price Index", secondary_y=True, row=2, col=1)
        
        fig.show()
    
    def analyze_stock(self, symbol):
        """Complete analysis with enhanced visualizations"""
        print(f"\n{'='*60}")
        print(f"🚀 ENHANCED SENTIMENT ANALYSIS FOR {symbol}")
        print(f"{'='*60}")
        
        print("📊 Collecting data...")
        headlines = self.scrape_news_headlines(symbol)
        sentiments = self.analyze_sentiment(headlines)
        stock_data = self.get_stock_data(symbol)
        
        if stock_data is None:
            print("❌ Error: Could not fetch stock data")
            return
        
        print("🎨 Creating Power BI-style visualizations...")
        
        # Create all dashboards
        self.create_executive_dashboard(sentiments, stock_data, symbol)
        time.sleep(1)
        self.create_sentiment_timeline_chart(sentiments, symbol)
        time.sleep(1)
        self.create_sentiment_kpi_dashboard(sentiments, symbol)
        time.sleep(1)
        self.create_correlation_analysis(sentiments, stock_data, symbol)
        
        # Display summary
        self.display_enhanced_results(symbol, sentiments, stock_data)
    
    def display_enhanced_results(self, symbol, sentiments, stock_data):
        """Display enhanced analysis results"""
        print(f"\n📈 EXECUTIVE SUMMARY FOR {symbol}")
        print("=" * 50)
        
        # Sentiment metrics
        scores = [s['score'] for s in sentiments]
        labels = [s['label'] for s in sentiments]
        
        avg_sentiment = np.mean(scores)
        sentiment_volatility = np.std(scores)
        
        positive_count = labels.count('Positive')
        negative_count = labels.count('Negative')
        neutral_count = labels.count('Neutral')
        
        print(f"📰 Headlines Analyzed: {len(sentiments)}")
        print(f"📊 Average Sentiment: {avg_sentiment:.3f}")
        print(f"📈 Sentiment Volatility: {sentiment_volatility:.3f}")
        print(f"✅ Positive: {positive_count} ({positive_count/len(sentiments)*100:.1f}%)")
        print(f"⚪ Neutral: {neutral_count} ({neutral_count/len(sentiments)*100:.1f}%)")
        print(f"❌ Negative: {negative_count} ({negative_count/len(sentiments)*100:.1f}%)")
        
        # Stock metrics
        current_price = stock_data['Close'].iloc[-1]
        price_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
        price_change_pct = (price_change / stock_data['Close'].iloc[-2]) * 100
        
        print(f"\n💰 STOCK PERFORMANCE")
        print("-" * 30)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Daily Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
        print(f"52W High: ${stock_data['High'].max():.2f}")
        print(f"52W Low: ${stock_data['Low'].min():.2f}")
        
        print(f"\n🎯 KEY INSIGHTS")
        print("-" * 20)
        if avg_sentiment > 0.2:
            print("🟢 Strong positive sentiment detected")
        elif avg_sentiment < -0.2:
            print("🔴 Strong negative sentiment detected")
        else:
            print("🟡 Neutral sentiment prevails")
        
        if sentiment_volatility > 0.3:
            print("⚡ High sentiment volatility - market uncertainty")
        else:
            print("📊 Stable sentiment pattern")

def main():
    analyzer = EnhancedStockSentimentAnalyzer()
    
    print("🚀 Enhanced Stock Sentiment Analyzer")
    print("💼 Power BI-Style Visualizations")
    print("=" * 50)
    
    print("\nSelect analysis type:")
    print("1. Single stock with full dashboard suite")
    print("2. Quick analysis")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    symbol = input("Enter stock symbol (e.g., AAPL): ").strip().upper()
    
    if not symbol:
        symbol = 'AAPL'
    
    if choice == '1':
        analyzer.analyze_stock(symbol)
    else:
        # Quick analysis without full dashboard
        headlines = analyzer.scrape_news_headlines(symbol)
        sentiments = analyzer.analyze_sentiment(headlines)
        stock_data = analyzer.get_stock_data(symbol)
        
        if stock_data is not None:
            analyzer.create_sentiment_kpi_dashboard(sentiments, symbol)
            analyzer.display_enhanced_results(symbol, sentiments, stock_data)
    
    print(f"\n✅ Analysis complete! Check your browser for Power BI-style dashboards.")

if __name__ == "__main__":
    main()
