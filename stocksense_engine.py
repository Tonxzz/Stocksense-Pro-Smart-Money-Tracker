import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures

class DataIngestion:
    def __init__(self, ticker=None, period="1y", interval="1d"):
        self.ticker = ticker
        self.period = period
        self.interval = interval

    def fetch_data(self, ticker_override=None):
        """Fetches data for a single ticker."""
        target_ticker = ticker_override if ticker_override else self.ticker
        if not target_ticker:
            raise ValueError("No ticker specified.")
            
        try:
            # Add random parameter to avoid cache if needed, but yf usually handles it.
            df = yf.download(target_ticker, period=self.period, interval=self.interval, progress=False, threads=False)
            
            if df.empty:
                return pd.DataFrame()

            # Flatten MultiIndex columns if present (common in new yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure required columns exist
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                return pd.DataFrame() 
                
            return df
        except Exception as e:
            print(f"Error fetching {target_ticker}: {e}")
            return pd.DataFrame()

    def check_safety_criteria(self, df):
        """
        Pre-screening Gate:
        1. Price > 60 IDR (Avoid Penny Stocks/Gocap)
        2. Liquidity > 2B IDR (Avg Vol 20D * Price)
        """
        if df.empty or len(df) < 20:
            return False, "Insufficient Data"
            
        last_close = df['close'].iloc[-1]
        avg_vol = df['volume'].iloc[-20:].mean()
        avg_liquidity = avg_vol * last_close
        
        if last_close <= 60:
            return False, f"Price too low ({last_close})"
            
        if avg_liquidity < 2_000_000_000: # 2 Billion IDR
            return False, f"Low Liquidity ({avg_liquidity/1e9:.2f}B)"
            
        return True, "Pass"

    # fetch_batch_latest removed to allow UI-controlled threading with progress bar in app.py

    def calculate_indicators(self, df):
        """Calculates Technical Indicators: RSI, BB, VWAP, RVOL, MFI, CMF."""
        df = df.copy()
        
        # 1. VWAP (Rolling approximation for daily, or intraday standard)
        # Using Typical Price * Volume / Cumulative Volume for the session is standard VWAP
        # For daily chart "session" is a year? No, usually rolling or anchored.
        # We will use "Rolling VWAP" (20 days) as a trend proxy for daily
        v = df['volume'].values
        tp = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (tp * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()

        # 2. Bollinger Bands (20, 2)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['std_20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2)
        df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2)
        
        # 3. RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 4. RVOL (Relative Volume)
        # Ratio of current volume to 20-day average volume
        df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
        df['rvol'] = df['volume'] / df['vol_ma_20']
        
        # 5. MFI (Money Flow Index)
        # Money Flow = TP * Vol
        # Positive Flow = Flow where current TP > prev TP
        # Negative Flow = Flow where current TP < prev TP
        raw_money_flow = tp * df['volume']
        
        positive_flow = np.where(tp > tp.shift(1), raw_money_flow, 0)
        negative_flow = np.where(tp < tp.shift(1), raw_money_flow, 0)
        
        positive_mf_sum = pd.Series(positive_flow, index=df.index).rolling(window=14).sum()
        negative_mf_sum = pd.Series(negative_flow, index=df.index).rolling(window=14).sum()
        
        mfi_ratio = positive_mf_sum / negative_mf_sum
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        df['mfi'] = df['mfi'].fillna(50) # Neutral fill
        
        # 6. CMF (Chaikin Money Flow) - 20 period
        # MFM = ((Close - Low) - (High - Close)) / (High - Low)
        # MFV = MFM * Volume
        # CMF = Sum(MFV, 20) / Sum(Vol, 20)
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0) # Handle 0 division if high==low
        mfv = mfm * df['volume']
        
        df['cmf'] = mfv.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
        df['cmf'] = df['cmf'].fillna(0)  # Handle division by zero
        
        # 7. SMA 200 (Major Trend Support)
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # 8. Fill NaN for critical columns to prevent chart errors
        df['rvol'] = df['rvol'].fillna(1.0)
        df['rsi'] = df['rsi'].fillna(50)
        df['vwap'] = df['vwap'].fillna(df['close'])
        
        return df

class SmartMoneyAnalyzer:
    """Interprets metrics to find Smart Money footprints."""
    
    @staticmethod
    def analyze_single_row(row):
        """
        Analyzes a single row using the Logic Patch v2.1:
        - RVOL > 1.5: +2 (Volume Explosion)
        - CMF > 0.15: +2 (Strong Acc) OR CMF > 0.05: +1 (Mod Acc)
        - Close > VWAP: +1 (Trend)
        - MFI < 30 & Close > VWAP: +1 (Dip Buy)
        """
        narrative = []
        score = 0
        
        # 1. Volume Explosion (Weighted heavily)
        if row['rvol'] > 1.5:
            narrative.append("VOLUME SPIKE (> 1.5x).")
            score += 2
        
        # 2. Smart Money Flow (CMF) - Tiered
        if row['cmf'] > 0.15:
            narrative.append("Strong Institutional Accumulation (CMF > 0.15).")
            score += 2
        elif row['cmf'] > 0.05:
            narrative.append("Moderate Inflow (CMF > 0.05).")
            score += 1
        elif row['cmf'] < -0.05:
             narrative.append("Distribution Detected (CMF < -0.05).")
            
        # 3. Trend Confirmation
        if row['close'] > row['vwap']:
            narrative.append("Bullish Trend > VWAP.")
            score += 1
        else:
            narrative.append("Price below Intraday Trend (VWAP).")
            
        # 4. Strategic Dip Buy (Oversold in Uptrend)
        if row['mfi'] < 30 and row['close'] > row['vwap']:
            narrative.append("Strategic Dip Buy (Oversold + Uptrend).")
            score += 1
        elif row['mfi'] > 75:
             narrative.append("Overbought Conditions.")

        # Cap max score at 5
        score = min(score, 5)

        # Final Decision Status based on Score
        if score >= 4:
            status = "STRONG BUY"
        elif score >= 2:
            status = "ACCUMULATION" 
        else:
            status = "NEUTRAL/WAIT"
        
        return {
            "status": status,
            "score": score,
            "details": " ".join(narrative) if narrative else "No significant anomalies."
        }
