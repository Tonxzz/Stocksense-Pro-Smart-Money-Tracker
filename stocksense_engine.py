"""
StockSense Pulse Pro - Core Engine v2.0
========================================
A complete rewrite of the stock analysis system for Indonesian Stock Market (IDX).

Architecture Upgrade:
- REPLACED: Facebook Prophet (poor stochastic volatility handling)
- NEW: XGBoost Binary Classification (predicts probability of >1% gain in 3 days)
- NEW: Smart Money Flow indicators (VWAP, MFI) as institutional activity proxy
- NEW: Relative Volume (RVOL) instead of log volume
- NEW: Time-decayed sentiment scoring

Author: StockSense Pulse Pro Team
Version: 2.0 (XGBoost + Technical Confluence Edition)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
import json

# XGBoost and sklearn for ML pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# ============================================================================
# CLASS 1: DATA INGESTION
# ============================================================================

class DataIngestion:
    """
    Handles fetching OHLCV data from Yahoo Finance and calculating technical indicators.
    
    Smart Money Proxy Philosophy:
    - Since we don't have real-time broker summary data, we use VWAP and MFI
      as proxies for institutional activity.
    - VWAP shows the average price weighted by volume - institutions benchmark here.
    - MFI combines price and volume to show money flow pressure.
    """
    
    def __init__(self, ticker: str, period: str = "1y"):
        """
        Initialize data ingestion for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'BBCA.JK' for Bank Central Asia)
            period: Historical data period (default: 1 year)
        """
        self.ticker = ticker
        self.period = period
        self.df = None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch OHLCV data from Yahoo Finance.
        
        Returns:
            DataFrame with OHLCV columns
        """
        print(f"[DataIngestion] Mengambil data {self.ticker} periode {self.period}...")
        
        stock = yf.Ticker(self.ticker)
        self.df = stock.history(period=self.period)
        
        if self.df.empty:
            raise ValueError(f"Tidak ada data tersedia untuk {self.ticker}")
        
        # Standardize column names (lowercase)
        self.df.columns = [col.lower() for col in self.df.columns]
        
        # Ensure we have required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print(f"[DataIngestion] Berhasil mengambil {len(self.df)} hari data trading")
        return self.df
    
    def calculate_vwap(self) -> pd.Series:
        """
        Calculate VWAP (Volume Weighted Average Price).
        
        Financial Engineering Logic:
        - VWAP = Σ(Typical Price × Volume) / Σ(Volume)
        - Typical Price = (High + Low + Close) / 3
        - VWAP adalah benchmark yang digunakan institusi untuk mengevaluasi
          kualitas eksekusi order mereka.
        - Harga di atas VWAP = bullish, di bawah VWAP = bearish
        
        Note: For daily data, we calculate a cumulative VWAP over a rolling window.
        """
        # Typical Price: fair value estimate for the day
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        
        # Use a 20-day rolling window for daily VWAP calculation
        # This represents approximately 1 month of trading
        window = 20
        
        cumulative_tp_vol = (typical_price * self.df['volume']).rolling(window=window).sum()
        cumulative_vol = self.df['volume'].rolling(window=window).sum()
        
        vwap = cumulative_tp_vol / cumulative_vol
        
        # VWAP Distance: how far price is from VWAP (normalized)
        # Positive = price above VWAP (institutional support), Negative = below
        self.df['vwap'] = vwap
        self.df['vwap_dist_pct'] = ((self.df['close'] - vwap) / vwap) * 100
        
        return vwap
    
    
    def calculate_cmf(self, period: int = 20) -> pd.Series:
        """
        Calculate Chaikin Money Flow (CMF).
        
        Financial Engineering Logic:
        - CMF mengukur akumulasi/distribusi institutional dalam periode tertentu (default 20 hari).
        - Fokus pada posisi close relatif terhadap high-low range, dikalikan volume.
        - Close di atas (High+Low)/2 dengan volume besar = Akumulasi Kuat.
        - Close di bawah (High+Low)/2 dengan volume besar = Distribusi Kuat.
        - Nilai > 0.1: Indikasi Buying Pressure (Smart Money In).
        - Nilai < -0.1: Indikasi Selling Pressure (Smart Money Out).
        """
        # Money Flow Multiplier
        # [(Close - Low) - (High - Close)] / (High - Low)
        # Sederhananya: (2*Close - High - Low) / (High - Low)
        mf_multiplier = ((2 * self.df['close']) - self.df['high'] - self.df['low']) / (self.df['high'] - self.df['low'])
        mf_multiplier = mf_multiplier.fillna(0)
        
        # Money Flow Volume
        mf_volume = mf_multiplier * self.df['volume']
        
        # 20-period Sum
        mf_volume_sum = mf_volume.rolling(window=period).sum()
        volume_sum = self.df['volume'].rolling(window=period).sum()
        
        # CMF
        cmf = mf_volume_sum / volume_sum
        cmf = cmf.fillna(0)
        
        self.df['cmf'] = cmf
        return cmf
    
    def calculate_mfi(self, period: int = 14) -> pd.Series:
        """
        Calculate MFI (Money Flow Index) - Smart Money Proxy indicator.
        
        Financial Engineering Logic:
        - MFI adalah RSI yang memperhitungkan volume, bukan hanya harga
        - Range: 0-100
        - MFI > 80: Overbought (money sudah masuk terlalu banyak, potensi koreksi)
        - MFI < 20: Oversold (money sudah keluar banyak, potensi reversal)
        - MFI 50-80 dengan harga naik = ACCUMULATION oleh institusi
        - MFI 20-50 dengan harga turun = DISTRIBUTION oleh institusi
        
        Formula:
        1. Typical Price = (High + Low + Close) / 3
        2. Raw Money Flow = Typical Price × Volume
        3. Positive MF: sum of RMF when TP > previous TP
        4. Negative MF: sum of RMF when TP < previous TP
        5. Money Ratio = Positive MF / Negative MF
        6. MFI = 100 - (100 / (1 + Money Ratio))
        """
        typical_price = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        raw_money_flow = typical_price * self.df['volume']
        
        # Determine if typical price increased or decreased
        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
        
        # Sum over the period
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Money Flow Ratio (handle division by zero)
        money_ratio = positive_mf / negative_mf.replace(0, np.nan)
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))
        mfi = mfi.fillna(50)  # Neutral if undefined
        
        self.df['mfi'] = mfi
        return mfi
    
    def calculate_rvol(self, window: int = 20) -> pd.Series:
        """
        Calculate RVOL (Relative Volume).
        
        Financial Engineering Logic:
        - RVOL = Volume Hari Ini / Rata-rata Volume 20 Hari
        - RVOL > 2.0: Volume sangat tinggi, kemungkinan ada aktivitas institusi
        - RVOL > 1.5: Volume di atas rata-rata, ada interest
        - RVOL < 0.5: Volume rendah, tidak ada interest, hindari trading
        
        PENTING: Kita TIDAK menggunakan Log Volume karena:
        - Log Volume kehilangan informasi magnitude yang penting
        - RVOL sudah normalized, bisa dibandingkan antar saham
        - RVOL langsung menunjukkan "berapa kali lipat dari normal"
        """
        avg_volume = self.df['volume'].rolling(window=window).mean()
        rvol = self.df['volume'] / avg_volume
        
        # Cap extreme values untuk menghindari outlier
        rvol = rvol.clip(upper=10)  # Max 10x normal volume
        
        self.df['rvol'] = rvol
        return rvol
    
    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate RSI (Relative Strength Index).
        
        Financial Engineering Logic:
        - RSI mengukur momentum: seberapa kuat buying vs selling pressure
        - RSI > 70: Overbought, potensi pullback
        - RSI < 30: Oversold, potensi bounce
        - RSI 40-60: Area netral, ikuti trend
        """
        delta = self.df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)
        
        self.df['rsi'] = rsi
        return rsi
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Financial Engineering Logic:
        - MACD Line = EMA(12) - EMA(26): Jarak antara momentum jangka pendek vs panjang
        - Signal Line = EMA(9) dari MACD Line: Trigger untuk entry/exit
        - Histogram = MACD - Signal: Kekuatan momentum
        
        Trading signals:
        - MACD cross above Signal = Bullish momentum building
        - MACD cross below Signal = Bearish momentum building
        - Histogram naik = Momentum makin kuat
        - Histogram turun = Momentum melemah
        """
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        self.df['macd'] = macd_line
        self.df['macd_signal'] = signal_line
        self.df['macd_hist'] = histogram
        
        return macd_line, signal_line, histogram
    
    def calculate_bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Financial Engineering Logic:
        - Bollinger Bands mengukur volatilitas dan menunjukkan zona overbought/oversold
        - Upper Band = SMA + (2 × StdDev)
        - Lower Band = SMA - (2 × StdDev)
        - Band Width: Semakin lebar = volatilitas tinggi, semakin sempit = konsolidasi
        
        Trading signals:
        - Price di Upper Band = Extended/Overbought
        - Price di Lower Band = Oversold, potensial bounce
        - Squeeze (bands menyempit) = Potensi breakout
        """
        sma = self.df['close'].rolling(window=period).mean()
        std = self.df['close'].rolling(window=period).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        self.df['bb_upper'] = upper_band
        self.df['bb_middle'] = sma
        self.df['bb_lower'] = lower_band
        
        # BB Position: Di mana harga relatif terhadap bands (0-100%)
        # 100% = di upper band, 0% = di lower band, 50% = di middle
        bb_position = ((self.df['close'] - lower_band) / (upper_band - lower_band)) * 100
        self.df['bb_position'] = bb_position.clip(0, 100)
        
        return upper_band, sma, lower_band
    
    def calculate_volatility(self, period: int = 20) -> pd.Series:
        """
        Calculate historical volatility (annualized).
        
        Financial Engineering Logic:
        - Volatility tinggi = Risiko tinggi, butuh stop loss lebih lebar
        - Volatility rendah = Market tenang, bisa pakai stop loss ketat
        - IDX stocks biasanya volatilitas 25-60% annually
        """
        log_returns = np.log(self.df['close'] / self.df['close'].shift(1))
        volatility = log_returns.rolling(window=period).std() * np.sqrt(252) * 100
        
        self.df['volatility'] = volatility
        return volatility
    
    def calculate_all_indicators(self) -> pd.DataFrame:
        """
        Calculate all technical indicators and return enriched DataFrame.
        """
        print("[DataIngestion] Menghitung indikator teknikal...")
        
        # Smart Money Proxy indicators
        self.calculate_vwap()
        self.calculate_mfi()
        self.calculate_cmf()
        
        # Volume indicator (normalized)
        self.calculate_rvol()
        
        # Traditional technical indicators
        self.calculate_rsi()
        self.calculate_macd()
        self.calculate_bollinger_bands()
        self.calculate_volatility()
        
        # Handle missing values
        # Forward fill first, then backfill for any remaining NaN at the start
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        # Drop any remaining NaN rows (usually first few rows)
        initial_len = len(self.df)
        self.df = self.df.dropna()
        dropped = initial_len - len(self.df)
        
        if dropped > 0:
            print(f"[DataIngestion] Dropped {dropped} rows dengan missing values")
        
        print(f"[DataIngestion] Total {len(self.df)} rows dengan {len(self.df.columns)} fitur")
        return self.df
    
    def get_feature_summary(self) -> Dict:
        """
        Return summary of latest indicator values.
        """
        if self.df is None or self.df.empty:
            return {}
        
        latest = self.df.iloc[-1]
        
        return {
            'ticker': self.ticker,
            'date': self.df.index[-1].strftime('%Y-%m-%d'),
            'close': round(latest['close'], 2),
            'volume': int(latest['volume']),
            'rvol': round(latest['rvol'], 2),
            'vwap': round(latest['vwap'], 2),
            'vwap_dist_pct': round(latest['vwap_dist_pct'], 2),
            'mfi': round(latest['mfi'], 2),
            'cmf': round(latest['cmf'], 3),
            'rsi': round(latest['rsi'], 2),
            'macd': round(latest['macd'], 4),
            'macd_hist': round(latest['macd_hist'], 4),
            'bb_position': round(latest['bb_position'], 2),
            'volatility': round(latest['volatility'], 2)
        }


# ============================================================================
# CLASS 2: SENTIMENT ANALYZER (RSS FEED ENGINE)
# ============================================================================

import feedparser
import urllib.parse
from datetime import datetime
import numpy as np
from typing import List, Dict

def get_company_name(ticker: str) -> str:
    """Map ticker to searchable company name."""
    mapping = {
        'BBCA.JK': 'Bank Central Asia',
        'BBRI.JK': 'Bank Rakyat Indonesia',
        'BMRI.JK': 'Bank Mandiri',
        'BBNI.JK': 'Bank Negara Indonesia',
        'TLKM.JK': 'Telkom Indonesia',
        'ASII.JK': 'Astra International',
        'ICBP.JK': 'Indofood CBP',
        'UNVR.JK': 'Unilever Indonesia',
        'GOTO.JK': 'GoTo Gojek Tokopedia',
        'ADRO.JK': 'Adaro Energy',
        'PTBA.JK': 'Bukit Asam',
        'PGAS.JK': 'Perusahaan Gas Negara',
        'ANTM.JK': 'Aneka Tambang',
        'BUMI.JK': 'Bumi Resources',
        'ENRG.JK': 'Energi Mega Persada',
        'BRIS.JK': 'Bank Syariah Indonesia',
        'BFIN.JK': 'BFI Finance'
    }
    return mapping.get(ticker, ticker.replace('.JK', ''))

class SentimentAnalyzer:
    """
    Live sentiment analysis using Google News RSS Feed (Most Stable).
    URL: https://news.google.com/rss/search?q={QUERY}+when:{DAYS}d&hl=id&gl=ID&ceid=ID:id
    """
    
    def __init__(self, decay_factor: float = 0.8):
        self.decay_factor = decay_factor
        
        # Simple Logic Keywords (Expanded for IDX context)
        self.bullish_keywords = [
            'laba', 'naik', 'melesat', 'dividen', 'akuisisi', 'untung', 'positif', 
            'buy', 'tumbuh', 'rekor', 'prospek cerah', 'target price naik', 
            'accumulate', 'outperform', 'ekspansi', 'kerjasama', 'merger', 
            'disetujui', 'divestasi', 'buyback', 'signifikan', 'menguat',
            'terkerek', 'melonjak', 'cuan', 'optimis'
        ]
        
        self.bearish_keywords = [
            'rugi', 'turun', 'anjlok', 'utang', 'gagal', 'negatif', 'sell', 
            'koreksi', 'lemah', 'tertekan', 'kolaps', 'pkpu', 'suspend', 
            'downgrade', 'underperform', 'peringatan', 'gugatan', 'pailit',
            'ambles', 'kebakaran', 'denda', 'phk', 'mundur'
        ]
    
    def fetch_live_news(self, ticker: str, days: int = 30) -> List[Dict]:
        """
        Fetch news using Google News RSS.
        """
        company_name = get_company_name(ticker)
        
        # Smart Query Construction: Ticker OR Company Name
        # Example: "ENRG.JK" OR "Energi Mega Persada"
        clean_ticker = ticker.replace('.JK', '')
        query = f'"{clean_ticker}" OR "{company_name}"'
        
        # URL Encode the query
        encoded_query = urllib.parse.quote(query)
        
        # Construct RSS URL
        # Docs: q={query}+when:{days}d
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}+when:{days}d&hl=id&gl=ID&ceid=ID:id"
        
        print(f"[Sentiment] Fetching RSS: {rss_url}")
        
        try:
            feed = feedparser.parse(rss_url)
            
            clean_news = []
            today = datetime.now()
            
            if not feed.entries:
                print("[Sentiment] No entries found in RSS feed.")
                return []
                
            print(f"[Sentiment] RSS returned {len(feed.entries)} items.")
            
            for item in feed.entries:
                title = item.get('title', 'No Title')
                link = item.get('link', '#')
                pub_date = item.get('published', '')
                source = item.get('source', {}).get('title', 'Google News')
                
                # Parse Date (RSS usually RFC-822)
                # Sat, 07 Jan 2026 10:00:00 GMT
                try:
                    # feedparser often parses 'published_parsed' to struct_time
                    if item.get('published_parsed'):
                        dt_object = datetime(*item.published_parsed[:6])
                        days_ago = (today - dt_object).days
                    else:
                        days_ago = 0 # Fallback
                except:
                    days_ago = 0
                
                # Cap days (just in case)
                days_ago = max(0, days_ago)
                
                # Google News RSS title often format: "Title - Source"
                # We can clean it if needed, but keeping it is fine.
                
                clean_news.append({
                    'text': title,
                    'link': link,
                    'media': source,
                    'date': pub_date[:16], # Shorten display
                    'days_ago': days_ago
                })
            
            return clean_news[:20] # Top 20 relevant
            
        except Exception as e:
            print(f"[Sentiment] RSS Fetch Error: {e}")
            return []

    def _analyze_headline(self, headline: str) -> float:
        """
        Simple Keyword Scoring.
        """
        headline_lower = headline.lower()
        score = 0.0
        
        # Check Bullish
        for kw in self.bullish_keywords:
            if kw in headline_lower:
                score += 0.5
        
        # Check Bearish
        for kw in self.bearish_keywords:
            if kw in headline_lower:
                score -= 0.5
        
        # Clip score
        return float(np.clip(score, -1.0, 1.0))
    
    def analyze_headlines(self, headlines: List[Dict]) -> Dict:
        """
        Analyze a list of headlines with time decay.
        """
        if not headlines:
            return {
                'individual_scores': [],
                'weighted_aggregate': 0.0,
                'raw_aggregate': 0.0,
                'sentiment_label': 'NEUTRAL',
                'headline_count': 0
            }
        
        individual_scores = []
        weighted_sum = 0.0
        weight_sum = 0.0
        
        for item in headlines:
            headline_text = item.get('text', '')
            days_ago = item.get('days_ago', 0)
            
            # Get raw sentiment score
            raw_score = self._analyze_headline(headline_text)
            
            # Apply time decay
            decay_weight = self.decay_factor ** days_ago
            weighted_score = raw_score * decay_weight
            
            individual_scores.append({
                'headline': headline_text,
                'link': item.get('link', '#'),
                'media': item.get('media', ''),
                'date_str': item.get('date', ''),
                'raw_score': raw_score,
                'days_ago': days_ago,
                'decay_weight': round(decay_weight, 3),
                'weighted_score': round(weighted_score, 3)
            })
            
            weighted_sum += weighted_score
            weight_sum += decay_weight
        
        # Calculate weighted average
        weighted_aggregate = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        raw_aggregate = sum(s['raw_score'] for s in individual_scores) / len(individual_scores)
        
        # Determine sentiment label
        if weighted_aggregate >= 0.15:  # Adjusted threshold
            label = 'BULLISH'
        elif weighted_aggregate <= -0.15:
            label = 'BEARISH'
        else:
            label = 'NEUTRAL'
        
        return {
            'individual_scores': individual_scores,
            'weighted_aggregate': round(weighted_aggregate, 3),
            'raw_aggregate': round(raw_aggregate, 3),
            'sentiment_label': label,
            'headline_count': len(headlines)
        }
    
    def get_mock_headlines(self, ticker: str) -> List[Dict]:
        return self.fetch_live_news(ticker)


# ============================================================================
# CLASS 3: QUANT MODEL (XGBoost Classifier)
# ============================================================================

class QuantModel:
    """
    XGBoost-based prediction model for stock price movement.
    
    Why XGBoost over Prophet?
    1. Prophet designed for time series with strong trends and seasonality (not volatile stocks)
    2. XGBoost better at capturing non-linear relationships in features
    3. Classification (will price go up?) is more actionable than regression (what will price be?)
    4. XGBoost outputs probability, enabling risk-calibrated decisions
    
    Target Variable:
    - Binary: Will price rise >1% in the next 3 trading days?
    - 1 = Yes (BUY signal territory)
    - 0 = No (WAIT or SELL signal territory)
    
    Why >1% threshold?
    - Filters out noise (small fluctuations)
    - >1% is a meaningful gain considering transaction costs
    - 3 days is short-term enough for swing trading, long enough to filter intraday noise
    """
    
    def __init__(self, lookahead_days: int = 3, threshold_pct: float = 1.0):
        """
        Initialize quant model.
        
        Args:
            lookahead_days: Number of days to look ahead for target calculation
            threshold_pct: Minimum % gain to be considered positive class
        """
        self.lookahead_days = lookahead_days
        self.threshold_pct = threshold_pct
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.metrics = {}
    
    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Create binary target variable: Will price rise >1% in next 3 days?
        
        Financial Engineering Logic:
        - Kita tidak prediksi harga absolut (sangat sulit dan tidak actionable)
        - Kita prediksi PROBABILITAS harga akan naik cukup signifikan
        - Threshold 1% memfilter noise dan memperhitungkan biaya transaksi
        """
        # Calculate future return (percentage change from today to N days ahead)
        future_close = df['close'].shift(-self.lookahead_days)
        future_return = ((future_close - df['close']) / df['close']) * 100
        
        # Binary target: 1 if return > threshold, else 0
        target = (future_return > self.threshold_pct).astype(int)
        
        # Name the series for clarity
        target.name = 'target'
        
        return target
    
    def prepare_features(self, df: pd.DataFrame, sentiment_score: float = 0.0) -> pd.DataFrame:
        """
        Prepare features for model training/prediction.
        
        Feature Categories:
        1. Smart Money Proxy: VWAP distance, MFI
        2. Volume: RVOL (relative volume)
        3. Momentum: RSI, MACD histogram
        4. Volatility: BB position, historical volatility
        5. Sentiment: Aggregated news sentiment score
        """
        features = pd.DataFrame(index=df.index)
        
        # Smart Money Proxy features
        features['vwap_dist'] = df['vwap_dist_pct']  # Distance from VWAP (%)
        features['mfi'] = df['mfi']  # Money Flow Index (0-100)
        
        # Volume feature (normalized)
        features['rvol'] = df['rvol']  # Relative Volume
        
        # Momentum features
        features['rsi'] = df['rsi']  # RSI (0-100)
        features['macd_hist'] = df['macd_hist']  # MACD Histogram
        
        # Volatility features
        features['bb_position'] = df['bb_position']  # Position in Bollinger Bands (0-100)
        features['volatility'] = df['volatility']  # Historical volatility (%)
        
        # Derived features for additional signal
        # RSI momentum: Rate of change of RSI
        features['rsi_momentum'] = df['rsi'].diff(5)
        
        # MFI trend: Is MFI increasing or decreasing?
        features['mfi_trend'] = df['mfi'].diff(5)
        
        # Price trend: 5-day return
        features['return_5d'] = ((df['close'] - df['close'].shift(5)) / df['close'].shift(5)) * 100
        
        # Volume trend
        features['rvol_trend'] = df['rvol'].rolling(5).mean()
        
        # Sentiment score (constant for all rows in current implementation)
        features['sentiment'] = sentiment_score
        
        # Fill NaN with forward fill then backfill
        features = features.fillna(method='ffill').fillna(method='bfill')
        
        # Replace any inf values
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        self.feature_columns = features.columns.tolist()
        
        return features
    
    def train(self, df: pd.DataFrame, sentiment_score: float = 0.0, 
              test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Train XGBoost classifier on historical data.
        
        Args:
            df: DataFrame with technical indicators
            sentiment_score: Aggregated sentiment score
            test_size: Proportion for test set
            random_state: Random seed for reproducibility
        
        Returns:
            Dict with training metrics
        """
        print("[QuantModel] Mempersiapkan data training...")
        
        # Create target
        target = self.create_target(df)
        
        # Prepare features
        features = self.prepare_features(df, sentiment_score)
        
        # Align target and features (remove last N rows where target is NaN)
        valid_idx = target.notna()
        X = features[valid_idx]
        y = target[valid_idx]
        
        print(f"[QuantModel] Total samples: {len(X)}, Positive class: {y.sum()} ({y.mean()*100:.1f}%)")
        
        # Train/test split (temporal for time series, but random works for demo)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize XGBoost with reasonable defaults for financial data
        self.model = XGBClassifier(
            n_estimators=100,       # Number of boosting rounds
            max_depth=4,            # Prevent overfitting on noisy financial data
            learning_rate=0.1,      # Standard learning rate
            subsample=0.8,          # Use 80% of data per tree (regularization)
            colsample_bytree=0.8,   # Use 80% of features per tree
            min_child_weight=3,     # Minimum samples in leaf (regularization)
            gamma=0.1,              # Regularization parameter
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        print("[QuantModel] Training XGBoost classifier...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.metrics = {
            'accuracy': round(accuracy_score(y_test, y_pred), 4),
            'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'positive_rate_train': round(y_train.mean(), 4),
            'positive_rate_test': round(y_test.mean(), 4)
        }
        
        print(f"[QuantModel] Training complete!")
        print(f"  - Accuracy: {self.metrics['accuracy']:.2%}")
        print(f"  - Precision: {self.metrics['precision']:.2%}")
        print(f"  - Recall: {self.metrics['recall']:.2%}")
        print(f"  - F1 Score: {self.metrics['f1_score']:.2%}")
        
        return self.metrics
    
    def predict(self, df: pd.DataFrame, sentiment_score: float = 0.0) -> Dict:
        """
        Predict probability for the latest data point.
        
        Returns:
            Dict with probability and prediction details
        """
        if self.model is None:
            raise ValueError("Model belum ditraining. Panggil train() terlebih dahulu.")
        
        # Prepare features for latest row
        features = self.prepare_features(df, sentiment_score)
        latest_features = features.iloc[[-1]]  # Get last row as DataFrame
        
        # Scale features
        latest_scaled = self.scaler.transform(latest_features)
        
        # Predict probability
        prob = self.model.predict_proba(latest_scaled)[0, 1]
        pred_class = self.model.predict(latest_scaled)[0]
        
        return {
            'probability': round(prob, 4),
            'predicted_class': int(pred_class),
            'interpretation': 'Likely to rise >1% in 3 days' if pred_class == 1 else 'Unlikely to rise >1% in 3 days',
            'confidence': round(max(prob, 1-prob), 4),
            'model_metrics': self.metrics
        }
    
    def get_feature_importance(self) -> Dict:
        """
        Return feature importance from trained model.
        """
        if self.model is None:
            return {}
        
        importance = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_columns, importance))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return sorted_importance



# ============================================================================
# CLASS 4: SMART MONEY SCREENER
# ============================================================================

class SmartMoneyScreener:
    """
    Screener engine to identify stocks with excessive institutional activity.
    Rank and filter based on RVOL (Relative Volume), MFI, and CMF.
    """
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        
    def scan(self, progress_callback=None) -> pd.DataFrame:
        """
        Scan the list of tickers and return a summary DataFrame.
        """
        results = []
        
        total = len(self.tickers)
        for i, ticker in enumerate(self.tickers):
            if progress_callback:
                progress_callback(i / total, f"Scanning {ticker}...")
                
            try:
                # Use a shorter period for speed, but enough for indicators (6mo is safe)
                engine = DataIngestion(ticker, period="6mo")
                # Suppress prints
                import sys, os
                original_stdout = sys.stdout
                sys.stdout = open(os.devnull, 'w')
                
                try:
                    df = engine.fetch_data()
                    df = engine.calculate_all_indicators()
                    summary = engine.get_feature_summary()
                    
                    # Determine Smart Money Status
                    rvol = summary['rvol']
                    mfi = summary['mfi']
                    cmf = summary['cmf']
                    close = summary['close']
                    vwap_dist = summary['vwap_dist_pct']
                    
                    status = "NEUTRAL"
                    
                    # Logic Deterministik Smart Money
                    if rvol > 1.5:
                        if mfi > 60 and cmf > 0.05:
                            status = "🔥 STRONG ACCUMULATION"
                        elif mfi < 40 and cmf < -0.05:
                            status = "❄️ STRONG DISTRIBUTION"
                        elif cmf > 0.1:
                            status = "🟢 INFLOW"
                        elif cmf < -0.1:
                            status = "🔴 OUTFLOW"
                        else:
                            status = "⚠️ HIGH VOL / CHURNING"
                    else:
                        if cmf > 0.15:
                            status = "🟢 STEADY ACCUMULATION"
                        elif cmf < -0.15:
                            status = "🔴 STEADY DISTRIBUTION"
                            
                    results.append({
                        'Ticker': ticker,
                        'Price': close,
                        'RVOL': rvol,
                        'MFI': mfi,
                        'CMF': cmf,
                        'VWAP Dist %': vwap_dist,
                        'Status': status
                    })
                    
                except Exception:
                    pass
                finally:
                    sys.stdout = original_stdout
                    
            except Exception as e:
                print(f"Error scanning {ticker}: {e}")
                
        if progress_callback:
            progress_callback(1.0, "Scan Complete!")
            
        return pd.DataFrame(results)

# ============================================================================
# CLASS 5: SIGNAL TRANSLATOR
# ============================================================================

class SignalTranslator:
    """
    Translates XGBoost probability + technical context into human-readable signals.
    
    This is the "Bridge" between quantitative model output and actionable trading decisions.
    
    Translation Logic:
    - Combines probability (ML model confidence) with current market context
    - Adjusts for volatility (high volatility = higher risk, need stronger signal)
    - Produces narrative text suitable for UI display
    
    Output Format:
    - Signal Strength: Strong/Moderate/Weak
    - Action: BUY/WAIT/SELL
    - Confidence: 0-100%
    - Risk Warning: Alert if conditions are risky
    """
    
    def __init__(self):
        """
        Initialize signal translator with thresholds.
        """
        # Probability thresholds for different signal strengths
        self.thresholds = {
            'strong_buy': 0.60,
            'buy': 0.40,
            'neutral': 0.20
        }
        
        # Technical confluence rules
        self.rules = {
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'mfi_strong_flow': 60,
            'mfi_weak_flow': 40,
            'cmf_strong_inflow': 0.1,
            'cmf_strong_outflow': -0.1,
            'volatility_high': 40,  # >40% annualized is high for IDX
            'rvol_high': 2.0,       # 2x normal volume
            'vwap_significant_dist': 2.0  # 2% above/below VWAP
        }
    
    def _get_technical_context(self, indicators: Dict) -> Dict:

        """
        Analyze current technical conditions.
        """
        context = {
            'trend_bias': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'volume_activity': 'NORMAL',
            'risk_level': 'MEDIUM',
            'confluence_score': 0
        }
        
        # Trend bias from VWAP distance
        vwap_dist = indicators.get('vwap_dist_pct', 0)
        if vwap_dist > self.rules['vwap_significant_dist']:
            context['trend_bias'] = 'BULLISH'
            context['confluence_score'] += 1
        elif vwap_dist < -self.rules['vwap_significant_dist']:
            context['trend_bias'] = 'BEARISH'
            context['confluence_score'] -= 1
        
        # Momentum from RSI and MFI
        rsi = indicators.get('rsi', 50)
        mfi = indicators.get('mfi', 50)
        
        if rsi > self.rules['rsi_overbought']:
            context['momentum'] = 'OVERBOUGHT'
            context['confluence_score'] -= 0.5  # Risk of reversal
        elif rsi < self.rules['rsi_oversold']:
            context['momentum'] = 'OVERSOLD'
            context['confluence_score'] += 0.5  # Potential bounce
        elif rsi > 55 and mfi > self.rules['mfi_strong_flow']:
            context['momentum'] = 'BULLISH'
            context['confluence_score'] += 1
        elif rsi < 45 and mfi < self.rules['mfi_weak_flow']:
            context['momentum'] = 'BEARISH'
            context['confluence_score'] -= 1
            
        # CMF Check (Additional Smart Money Confirmation)
        cmf = indicators.get('cmf', 0)
        if cmf > self.rules['cmf_strong_inflow']:
            context['confluence_score'] += 0.5
        elif cmf < self.rules['cmf_strong_outflow']:
            context['confluence_score'] -= 0.5
        
        # Volume activity
        rvol = indicators.get('rvol', 1.0)
        if rvol > self.rules['rvol_high']:
            context['volume_activity'] = 'HIGH'
            context['confluence_score'] += 0.5  # Institutional interest
        elif rvol < 0.5:
            context['volume_activity'] = 'LOW'
            context['confluence_score'] -= 0.3  # No interest, avoid
        
        # Risk level from volatility
        vol = indicators.get('volatility', 25)
        if vol > self.rules['volatility_high']:
            context['risk_level'] = 'HIGH'
        elif vol < 20:
            context['risk_level'] = 'LOW'
        
        return context
    
    def _generate_narrative(self, indicators: Dict, context: Dict, 
                           probability: float, sentiment_label: str) -> str:
        """
        Generate human-readable technical narrative.
        """
        narratives = []
        
        # Price vs VWAP
        vwap_dist = indicators.get('vwap_dist_pct', 0)
        if vwap_dist > 0:
            narratives.append(f"Harga trading {abs(vwap_dist):.1f}% DI ATAS VWAP (bullish positioning)")
        else:
            narratives.append(f"Harga trading {abs(vwap_dist):.1f}% DI BAWAH VWAP (bearish positioning)")
        
        # RSI condition
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            narratives.append(f"RSI {rsi:.0f} - OVERBOUGHT, potensi pullback")
        elif rsi < 30:
            narratives.append(f"RSI {rsi:.0f} - OVERSOLD, potensi bounce")
        else:
            narratives.append(f"RSI {rsi:.0f} - momentum {'bullish' if rsi > 55 else 'bearish' if rsi < 45 else 'netral'}")
        
        # MFI Condition
        mfi = indicators.get('mfi', 50)
        if mfi > 60:
            narratives.append(f"MFI {mfi:.0f} - Smart Money ACCUMULATION detected")
        elif mfi < 40:
            narratives.append(f"MFI {mfi:.0f} - Smart Money DISTRIBUTION detected")
            
        # CMF
        cmf = indicators.get('cmf', 0)
        if cmf > 0.1:
            narratives.append(f"CMF {cmf:.2f} (Positif) - Tekanan Beli Konsisten")
        elif cmf < -0.1:
            narratives.append(f"CMF {cmf:.2f} (Negatif) - Tekanan Jual Konsisten")
        
        # Volume
        rvol = indicators.get('rvol', 1.0)
        if rvol > 2:
            narratives.append(f"Volume {rvol:.1f}x normal - AKTIVITAS INSTITUSI tinggi")
        elif rvol < 0.5:
            narratives.append(f"Volume rendah ({rvol:.1f}x) - minat pasar rendah")
        
        # Sentiment
        if sentiment_label not in ['NEUTRAL']:
            narratives.append(f"Sentimen berita: {sentiment_label}")
        
        # ML model insight
        if probability > 0.6:
            narratives.append(f"Model ML: {probability*100:.0f}% probabilitas naik >1% dalam 3 hari")
        elif probability < 0.4:
            narratives.append(f"Model ML: Rendah ({probability*100:.0f}%) probabilitas kenaikan signifikan")
        
        return " | ".join(narratives)
    
    def _generate_risk_warnings(self, indicators: Dict, context: Dict) -> List[str]:
        """
        Generate risk warnings based on current conditions.
        """
        warnings = []
        
        # High volatility warning
        vol = indicators.get('volatility', 25)
        if vol > self.rules['volatility_high']:
            warnings.append(f"⚠️ VOLATILITAS TINGGI ({vol:.0f}%): Gunakan stop loss lebih lebar")
        
        # Overbought warning
        rsi = indicators.get('rsi', 50)
        if rsi > 70:
            warnings.append("⚠️ OVERBOUGHT: Risiko pullback tinggi, hindari chase")
        
        # Low volume warning
        rvol = indicators.get('rvol', 1.0)
        if rvol < 0.5:
            warnings.append("⚠️ VOLUME RENDAH: Likuiditas terbatas, spread mungkin lebar")
        
        # Below VWAP warning
        vwap_dist = indicators.get('vwap_dist_pct', 0)
        if vwap_dist < -3:
            warnings.append("⚠️ JAUH DI BAWAH VWAP: Tekanan jual institusional")
        
        return warnings
    
    def translate(self, probability: float, indicators: Dict, sentiment_result: Dict) -> Dict:
        """
        Main translation function.
        
        Args:
            probability: XGBoost model probability (0.0 to 1.0)
            indicators: Dict of current technical indicators
            sentiment_result: Dict from SentimentAnalyzer
        
        Returns:
            Dict with complete signal translation in JSON format
        """
        # Get technical context
        context = self._get_technical_context(indicators)
        
        # Determine base signal from probability
        if probability >= self.thresholds['strong_buy']:
            base_signal = 'STRONG'
            base_action = 'BUY'
        elif probability >= self.thresholds['buy']:
            base_signal = 'MODERATE'
            base_action = 'BUY'
        elif probability >= self.thresholds['neutral']:
            base_signal = 'STABLE'
            base_action = 'HOLD'
        else:
            base_signal = 'WEAK'
            base_action = 'SELL'
        
        # Adjust for technical confluence
        confluence = context['confluence_score']
        
        # Special Logic: Hard Sell Override
        # If Price < VWAP AND MFI < 20 (Distribution), Force Sell even if prob is OK
        vwap_dist = indicators.get('vwap_dist_pct', 0)
        mfi = indicators.get('mfi', 50)
        
        if vwap_dist < 0 and mfi < 20:
            base_action = 'SELL'
            base_signal = 'STRONG'

        # Generate narratives
        sentiment_label = sentiment_result.get('sentiment_label', 'NEUTRAL')
        narrative = self._generate_narrative(indicators, context, probability, sentiment_label)
        risk_warnings = self._generate_risk_warnings(indicators, context)
        
        # Custom Recommendation for Hold
        recommendation_text = self._get_recommendation_text(base_signal, base_action, context)
        if base_action == 'HOLD':
            narrative += " | Trend Masih Kuat / Konsolidasi"
        
        # Compile final signal
        signal = {
            'signal_strength': base_signal,
            'action': base_action,
            'confidence_pct': round(probability * 100, 1),
            'ml_probability': round(probability, 4),
            'technical_context': context,
            'sentiment_summary': {
                'label': sentiment_label,
                'weighted_score': sentiment_result.get('weighted_aggregate', 0.0)
            },
            'narrative': narrative,
            'risk_warnings': risk_warnings,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'recommendation': self._get_recommendation_text(base_signal, base_action, context)
        }
        
        return signal
    
    def _get_recommendation_text(self, signal: str, action: str, context: Dict) -> str:
        """
        Generate final recommendation text.
        """
        if action == 'BUY':
            if signal == 'STRONG':
                return "REKOMENDASI: Pertimbangkan entry dengan ukuran posisi penuh. Confluence teknikal mendukung."
            else:
                return "REKOMENDASI: Pertimbangkan entry dengan ukuran posisi moderat. Wait for confirmation preferred."
        elif action == 'HOLD':
            return "REKOMENDASI: Trend Masih Kuat / Konsolidasi. Hold posisi yang ada. Jangan entry agresif."
        elif action == 'SELL':
            return "REKOMENDASI: Pertimbangkan exit atau reduce posisi. Tekanan jual dominan."
        else:
            return "REKOMENDASI: Tunggu setup yang lebih baik."


# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================

def main():
    """
    End-to-end demonstration of StockSense Pulse Pro engine.
    """
    print("=" * 70)
    print("STOCKSENSE PULSE PRO - CORE ENGINE v2.0")
    print("XGBoost Classification + Smart Money Flow Analysis")
    print("=" * 70)
    print()
    
    # Configuration
    TICKER = "BBCA.JK"  # Bank Central Asia (most liquid IDX stock)
    PERIOD = "1y"       # 1 year of historical data
    
    try:
        # ====================================================================
        # STEP 1: DATA INGESTION
        # ====================================================================
        print("\n" + "=" * 50)
        print("STEP 1: DATA INGESTION")
        print("=" * 50)
        
        data_engine = DataIngestion(ticker=TICKER, period=PERIOD)
        df = data_engine.fetch_data()
        df = data_engine.calculate_all_indicators()
        
        # Show latest indicator summary
        summary = data_engine.get_feature_summary()
        print("\n[Latest Indicator Summary]")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # ====================================================================
        # STEP 2: SENTIMENT ANALYSIS
        # ====================================================================
        print("\n" + "=" * 50)
        print("STEP 2: SENTIMENT ANALYSIS (Live Google News RSS)")
        print("=" * 50)
        
        sentiment_engine = SentimentAnalyzer(decay_factor=0.8)
        live_headlines = sentiment_engine.fetch_live_news(TICKER, days=30)
        sentiment_result = sentiment_engine.analyze_headlines(live_headlines)
        
        print("\n[Sentiment Analysis Result]")
        print(f"  Headlines analyzed: {sentiment_result['headline_count']}")
        print(f"  Raw aggregate score: {sentiment_result['raw_aggregate']}")
        print(f"  Weighted aggregate (with decay): {sentiment_result['weighted_aggregate']}")
        print(f"  Sentiment label: {sentiment_result['sentiment_label']}")
        
        print("\n[Individual Headlines with Decay]")
        for item in sentiment_result['individual_scores']:
            print(f"  * {item['headline']}")
            print(f"    Score: {item['raw_score']} x Decay({item['decay_weight']}) = {item['weighted_score']}")
        
        # ====================================================================
        # STEP 3: QUANT MODEL (XGBoost)
        # ====================================================================
        print("\n" + "=" * 50)
        print("STEP 3: QUANT MODEL (XGBoost Classification)")
        print("=" * 50)
        
        quant_model = QuantModel(lookahead_days=3, threshold_pct=1.0)
        
        # Train model
        metrics = quant_model.train(
            df=df, 
            sentiment_score=sentiment_result['weighted_aggregate'],
            test_size=0.2
        )
        
        # Get feature importance
        print("\n[Feature Importance]")
        importance = quant_model.get_feature_importance()
        for i, (feature, imp) in enumerate(importance.items()):
            if i < 5:  # Top 5 features
                print(f"  {feature}: {imp:.4f}")
        
        # Predict for current state
        prediction = quant_model.predict(df, sentiment_result['weighted_aggregate'])
        
        print("\n[Prediction for Current State]")
        print(f"  Probability of >1% gain in 3 days: {prediction['probability']*100:.1f}%")
        print(f"  Predicted class: {prediction['predicted_class']}")
        print(f"  Interpretation: {prediction['interpretation']}")
        
        # ====================================================================
        # STEP 4: SIGNAL TRANSLATION
        # ====================================================================
        print("\n" + "=" * 50)
        print("STEP 4: AI SIGNAL TRANSLATOR")
        print("=" * 50)
        
        translator = SignalTranslator()
        signal = translator.translate(
            probability=prediction['probability'],
            indicators=summary,
            sentiment_result=sentiment_result
        )
        
        # Pretty print signal
        print("\n" + "-" * 50)
        print("[FINAL TRADING SIGNAL]")
        print("-" * 50)
        print(f"  Ticker: {TICKER}")
        print(f"  Signal Strength: {signal['signal_strength']}")
        print(f"  Action: {signal['action']}")
        print(f"  Confidence: {signal['confidence_pct']}%")
        print(f"  Timestamp: {signal['timestamp']}")
        
        print("\n[Technical Context]:")
        for key, value in signal['technical_context'].items():
            print(f"  {key}: {value}")
        
        print("\n[Sentiment]:")
        print(f"  Label: {signal['sentiment_summary']['label']}")
        print(f"  Score: {signal['sentiment_summary']['weighted_score']}")
        
        print("\n[Narrative]:")
        print(f"  {signal['narrative']}")
        
        if signal['risk_warnings']:
            print("\n[Risk Warnings]:")
            for warning in signal['risk_warnings']:
                print(f"  {warning}")
        
        print("\n[Recommendation]:")
        print(f"  {signal['recommendation']}")
        
        # ====================================================================
        # STEP 5: EXPORT JSON (for UI consumption)
        # ====================================================================
        print("\n" + "=" * 50)
        print("STEP 5: JSON OUTPUT (for UI Integration)")
        print("=" * 50)
        
        print("\n[Signal JSON]")
        print(json.dumps(signal, indent=2, ensure_ascii=False, cls=NumpyEncoder))
        
        print("\n" + "=" * 70)
        print("STOCKSENSE PULSE PRO - Analysis Complete")
        print("=" * 70)
        
        return signal
        
    except Exception as e:
        print(f"\n[X] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
