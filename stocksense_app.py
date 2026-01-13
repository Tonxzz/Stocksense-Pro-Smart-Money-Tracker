"""
STOCKSENSE PULSE PRO v2 - INSTITUTIONAL DARK MODE
=================================================
A high-fidelity implementation of the "Institutional Dark Mode" UI.
Tech Stack: Streamlit, Plotly Graph Objects, Custom CSS Injection.

To run: streamlit run app_pro.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import Core Engine
try:
    from stocksense_engine import DataIngestion, SentimentAnalyzer, QuantModel, SignalTranslator
except ImportError:
    st.error("❌ Critical Error: 'stocksense_engine.py' not found.")
    st.stop()

# ============================================================================
# 1. PAGE CONFIG & CUSTOM CSS (THEME INJECTION)
# ============================================================================
st.set_page_config(
    page_title="StockSense Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# COLOR PALETTE
COLORS = {
    'bg_main': '#0E1117',
    'bg_card': '#1E232F',
    'text_primary': '#FFFFFF',
    'text_secondary': '#9CA3AF',
    'accent_red': '#FF4B4B',
    'accent_green': '#00C853',
    'accent_warning': '#FFA000',
    'grid_color': '#2C3342'
}

# CSS INJECTION
st.markdown(f"""
<style>
    /* Remove default top padding */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
    }}
    
    /* Main Background */
    .stApp {{
        background-color: {COLORS['bg_main']};
    }}
    
    /* Sidebar Background */
    section[data-testid="stSidebar"] {{
        background-color: #161B22;
    }}
    
    /* Input Fields */
    .stTextInput input, .stDateInput input {{
        background-color: #0d1117;
        color: white;
        border: 1px solid #30363d;
    }}
    
    /* Custom Card Styling */
    .st-card {{
        background-color: {COLORS['bg_card']};
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
        border: 1px solid #30363d;
        height: 100%;
    }}
    
    /* Signal Action Card Styling */
    .action-card {{
        border-radius: 12px;
        padding: 30px;
        text-align: center;
        color: white;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }}
    
    .action-title {{
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 10px;
        opacity: 0.9;
    }}
    
    .action-value {{
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }}
    
    .action-subtitle {{
        margin-top: 15px;
        font-size: 0.9rem;
        font-weight: 500;
        background: rgba(0,0,0,0.2);
        padding: 4px 12px;
        border-radius: 20px;
    }}
    
    /* Narrative Text */
    .narrative-text {{
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        line-height: 1.6;
        color: #E6EDF3;
    }}
    
    /* System Status Box in Sidebar */
    .system-status {{
        background: #1d3b58; /* Blue-ish */
        color: #58a6ff;
        padding: 12px;
        border-radius: 6px;
        font-size: 0.8rem;
        border: 1px solid #1f6feb;
        margin-top: 20px;
    }}
    
    /* Risk Banner */
    .risk-banner {{
        background-color: rgba(255, 75, 75, 0.15);
        border: 1px solid {COLORS['accent_red']};
        color: {COLORS['accent_red']};
        padding: 12px 20px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 20px;
    }}
    
</style>
""", unsafe_allow_html=True)


# ============================================================================
# 2. ADVANCED CHARTING (TRADINGVIEW STYLE SUBPLOTS)
# ============================================================================

def plot_advanced_charts(df, ticker, indicators):
    """
    Creates a 4-row subplot chart mimicking professional trading terminals.
    Row 1: Price + VWAP + BB (50% Height)
    Row 2: RSI (15% Height)
    Row 3: MFI (15% Height)
    Row 4: RVOL (20% Height)
    """
    
    # Calculate row heights
    specs = [
        [{"secondary_y": False}], # Row 1
        [{"secondary_y": False}], # Row 2
        [{"secondary_y": False}], # Row 3
        [{"secondary_y": False}]  # Row 4
    ]
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        specs=specs,
        subplot_titles=("", "", "", "") # Empty titles to save space, will use layout annotations if needed
    )
    
    # --- ROW 1: MAIN PRICE + CONFLUENCE ---
    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'],
        low=df['low'], close=df['close'],
        name='Price',
        increasing_line_color=COLORS['accent_green'],
        decreasing_line_color=COLORS['accent_red']
    ), row=1, col=1)
    
    # VWAP
    fig.add_trace(go.Scatter(
        x=df.index, y=df['vwap'],
        mode='lines', name='VWAP',
        line=dict(color=COLORS['accent_warning'], width=1.5)
    ), row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_upper'],
        mode='lines', name='BB Upper',
        line=dict(color='gray', width=1, dash='dot'),
        opacity=0.3, showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['bb_lower'],
        mode='lines', name='Band Fill',
        line=dict(color='gray', width=1, dash='dot'),
        opacity=0.3, fill='tonexty', fillcolor='rgba(119, 136, 153, 0.1)',
        showlegend=False
    ), row=1, col=1)
    
    # --- ROW 2: RSI ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df['rsi'],
        mode='lines', name='RSI',
        line=dict(color='#58a6ff', width=1.5)
    ), row=2, col=1)
    
    # Levels 70/30
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70, 
                 line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"), row=2, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30, 
                 line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"), row=2, col=1)
    
    # --- ROW 3: MFI ---
    fig.add_trace(go.Scatter(
        x=df.index, y=df['mfi'],
        mode='lines', name='MFI (Smart Money)',
        line=dict(color='#d2a8ff', width=1.5) # Light purple
    ), row=3, col=1)
    
    # Levels 80/20
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=80, y1=80, 
                 line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"), row=3, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=20, y1=20, 
                 line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"), row=3, col=1)

    # --- ROW 4: RVOL ---
    # Color logic: Yellow if > 2.0 (Anomaly), Grey otherwise
    colors = [COLORS['accent_warning'] if v >= 2.0 else '#4b5563' for v in df['rvol']]
    
    fig.add_trace(go.Bar(
        x=df.index, y=df['rvol'],
        name='RVOL',
        marker_color=colors
    ), row=4, col=1)
    
    # RVOL Baseline 1.0
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=1.0, y1=1.0, 
                 line=dict(color="rgba(255,255,255,0.5)", width=1), row=4, col=1)
    
    # --- LAYOUT STYLING ---
    fig.update_layout(
        template='plotly_dark',
        height=700,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=COLORS['bg_main'],
        paper_bgcolor=COLORS['bg_main'],
        showlegend=False,
    )
    
    # Remove gridlines for cleaner look
    fig.update_xaxes(showgrid=False, zeroline=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor='#1F2329', zeroline=False)
    
    # Axis labels on appropriate rows
    fig.update_yaxes(title_text=f"{ticker} Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MFI", row=3, col=1)
    fig.update_yaxes(title_text="RVOL", row=4, col=1)
    
    return fig

def plot_confidence_gauge(probability):
    """
    Gauge Chart for Probability (Confidence).
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probability * 100,
        title = {'text': "AI Confidence This Week", 'font': {'size': 14, 'color': '#9CA3AF'}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"}, # Invisible bar, we use steps for color
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 30], 'color': COLORS['accent_red']},
                {'range': [30, 60], 'color': COLORS['accent_warning']},
                {'range': [60, 100], 'color': COLORS['accent_green']}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)", 
        font = {'color': "white", 'family': "Arial"}, 
        margin=dict(l=30, r=30, t=50, b=20), 
        height=200
    )
    return fig

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # ------------------
    # 1. SIDEBAR
    # ------------------
    with st.sidebar:
        st.subheader("🎛️ Control Panel")
        
        # Ticker Input
        ticker = st.text_input("Ticker Symbol", value="ENRG.JK").upper()
        
        # Date Picker with Today Toggle
        col_toggle, col_empty = st.columns([1, 0.5])
        today_toggle = col_toggle.toggle("📅 Today's Data Only", value=False)
        
        today = datetime.now()
        
        if today_toggle:
             start_date = today
             end_date = today
             st.caption("Mode: Intraday Analysis")
        else:
             # Default to 3 months as requested
             default_start = today - timedelta(days=90)
             c1, c2 = st.columns(2)
             start_date = c1.date_input("Start Date", default_start)
             end_date = c2.date_input("End Date", today)
        
        # Calculate days for news fetching
        analysis_days = (end_date - start_date).days
        if analysis_days < 1: analysis_days = 1 # Minimum 1 day
        if analysis_days > 90: analysis_days = 90 # Max cap for reliability? Or left open.
        # User asked for "news analysis date for 3 months range" -> so we pass this to fetcher.
        
        st.write("") # Spacer
        
        # Action Button (Primary)
        if st.button("🚀 Analyze Market Pulse", type="primary", use_container_width=True):
            st.session_state['run_analysis'] = True
        
        # System Info Box
        st.markdown("""
        <div class="system-status">
            <strong>System: StockSense Pulse Pro</strong><br>
            <span style="opacity:0.8">Pro v2 Engine: XGBoost + Smart Money Flow</span><br>
            <span style="color:#00C853">● Active</span>
        </div>
        """, unsafe_allow_html=True)

    # ------------------
    # 2. ANALYSIS EXECUTION
    # ------------------
    if 'run_analysis' not in st.session_state:
        st.session_state['run_analysis'] = False
        
    if not st.session_state['run_analysis']:
        # Initial State
        st.title("StockSense Pro v2")
        st.info("Ready to analyze. Enter ticker and click 'Analyze Market Pulse' in the sidebar.")
        return

    # Run Analysis
    try:
        # Mocking or running real engine? Based on prompt "Gunakan kode backend"
        # We will run the real engine
        
        with st.spinner("Processing Market Data..."):
            # 1. Pipeline Execution
            data_engine = DataIngestion(ticker, "1y") 
            df = data_engine.fetch_data()
            df = data_engine.calculate_all_indicators()
            summary = data_engine.get_feature_summary()
            
            # Fetch Live Headlines (Now uses DuckDuckGo with Dynamic Date Range)
            sentiment_engine = SentimentAnalyzer()
            headlines = sentiment_engine.fetch_live_news(ticker, days=analysis_days)
            sentiment_result = sentiment_engine.analyze_headlines(headlines)
            
            quant_model = QuantModel(lookahead_days=3)
            quant_model.train(df, sentiment_result['weighted_aggregate'])
            prediction = quant_model.predict(df, sentiment_result['weighted_aggregate'])
            
            translator = SignalTranslator()
            signal = translator.translate(prediction['probability'], summary, sentiment_result)

        # ------------------
        # 3. SIGNAL DECK (TOP ROW)
        # ------------------
        st.markdown("### 🔴 AI Signal Translator")

        col1, col2, col3 = st.columns([1.5, 3, 1.5])
        
        # --- COL 1: ACTION CARD ---
        action = signal['action'] # BUY / SELL / WAIT
        strength = signal['signal_strength'] # STRONG / WEAK
        
        # Determine Color
        if action == "BUY":
            card_color = COLORS['accent_green']
        elif action == "SELL":
            card_color = COLORS['accent_red']
        elif action == "HOLD":
            card_color = COLORS['accent_warning'] # Amber for HOLD
        else:
            card_color = '#555555' # Grey for WAIT
            
        with col1:
            st.markdown(f"""
            <div class="action-card" style="background-color: {card_color};">
                <div class="action-title">Recommended Action</div>
                <div class="action-value">{action}</div>
                <div class="action-subtitle">Strength: {strength}</div>
            </div>
            """, unsafe_allow_html=True)
            
        # --- COL 2: NARRATIVE CONSOLE ---
        narrative_raw = signal['narrative']
        # Format narrative with bolding keywords (simple regex replacement or string ops)
        formatted_narrative = narrative_raw \
            .replace("DI ATAS VWAP", "<strong>DI ATAS VWAP</strong>") \
            .replace("DI BAWAH VWAP", "<strong>DI BAWAH VWAP</strong>") \
            .replace("ACCUMULATION", "<span style='color:#00C853'>ACCUMULATION</span>") \
            .replace("DISTRIBUTION", "<span style='color:#FF4B4B'>DISTRIBUTION</span>") \
            .replace("BULLISH", "<span style='color:#00C853'>BULLISH</span>") \
            .replace("BEARISH", "<span style='color:#FF4B4B'>BEARISH</span>")
            
        with col2:
            st.markdown(f"""
            <div class="st-card">
                <div style="color: #9CA3AF; font-size: 0.9rem; margin-bottom: 10px;">🤖 AI Narrative:</div>
                <div class="narrative-text">
                    {formatted_narrative}
                </div>
                <div style="margin-top: 20px; font-size: 0.8rem; color: #9CA3AF; border-top: 1px solid #30363d; padding-top: 10px;">
                    {signal['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # --- COL 3: GAUGE ---
        with col3:
            # We can put gauge inside a card or just raw
            # Let's put in card for consistency
            with st.container():
                # Hack to center via CSS? Plotly handles centering mostly
                st.plotly_chart(plot_confidence_gauge(signal['ml_probability']), use_container_width=True)

        # ------------------
        # 4. RISK BANNER
        # ------------------
        if signal['risk_warnings']:
            st.markdown(f"""
            <div class="risk-banner">
                ⚠️ High Risk Warning: &nbsp; {', '.join(signal['risk_warnings']).upper()}
            </div>
            """, unsafe_allow_html=True)
        else:
             # Spacer if no risk
             st.write("")

        # ------------------
        # 5. TECHNICAL DEEP DIVE (CHARTS)
        # ------------------
        st.markdown("### 🔎 Technical & Data Deep Dive")
        
        tab1, tab2, tab3, tab4 = st.tabs(["📉 Price & Confluence", "🧪 Feature Importance (XAI)", "📰 Sentiment & News", "📚 Panduan Chart"])
        
        with tab1:
            st.plotly_chart(plot_advanced_charts(df, ticker, summary), use_container_width=True)
            st.caption(f"{ticker} Price Structure | Orange Line = VWAP | Shaded = BB | Subplots: RSI, MFI, RVOL")
            
        with tab2:
            # Colorful Feature Importance (Restored)
            importance = quant_model.get_feature_importance()
            df_imp = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            df_imp = df_imp.sort_values('Importance', ascending=True).tail(10)
            
            fig_imp = px.bar(
                df_imp, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                color='Importance',
                color_continuous_scale='Viridis',
                title='XGBoost Feature Attribution'
            )
            
            fig_imp.update_layout(
                template='plotly_dark', 
                paper_bgcolor=COLORS['bg_main'],
                plot_bgcolor=COLORS['bg_main'],
                height=400,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_imp, use_container_width=True)

        with tab3:
            st.subheader(f"Sentiment Score: {sentiment_result['weighted_aggregate']} ({sentiment_result['sentiment_label']})")
            st.markdown("Analysis of latest headlines from **Google News RSS** (30 Days):")
            
            if not sentiment_result['individual_scores']:
                st.warning("No relevant news found for this ticker in the last 30 days.")
            else:
                for item in sentiment_result['individual_scores']:
                    # Color score for visibility
                    score_color = "#00C853" if item['raw_score'] > 0 else "#FF4B4B" if item['raw_score'] < 0 else "#888"
                    
                    st.markdown(f"""
                    <div style="background-color: #161B22; padding: 15px; border-radius: 8px; margin-bottom: 10px; border: 1px solid #30363d;">
                        <a href="{item['link']}" target="_blank" style="text-decoration: none; color: #58a6ff; font-weight: 600; font-size: 1.05em;">
                            🔗 {item['headline']}
                        </a>
                        <div style="margin-top: 5px; font-size: 0.85em; color: #8b949e;">
                            {item['media']} | {item['date_str']} ({item['days_ago']} days ago)
                        </div>
                        <div style="margin-top: 5px; font-size: 0.85em;">
                            Raw Score: <span style="color: {score_color}; font-weight: bold;">{item['raw_score']}</span> | 
                            Weighted Impact: <strong>{item['weighted_score']}</strong> (Decay: {item['decay_weight']})
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab4:
            st.markdown("""
            ### 📈 Panduan Interpretasi Chart
            
            **1. Panel Utama: Price & VWAP**
            - **Candlestick**: Pergerakan harga.
            - **🟠 Garis Oranye (VWAP)**: Level acuan institusi.
                - *Price > VWAP*: Bullish (Tren indikasi dikuasai pembeli).
                - *Price < VWAP*: Bearish (Tren indikasi dikuasai penjual).
            - **☁️ Area Shaded (Bollinger Bands)**: Mengukur volatilitas. Squeeze = Potensi ledakan.
            
            **2. Panel RSI (Momentum)**
            - **> 70 (Overbought)**: Hati-hati koreksi.
            - **< 30 (Oversold)**: Potensi rebound.
            
            **3. Panel MFI (Smart Money)**
            - Mirip RSI tapi pakai Volume.
            - Cari **Divergence**: Harga turun tapi MFI naik = Akumulasi diam-diam.
            
            **4. Panel RVOL (Volume Spike)**
            - **> 1.5**: Aktivitas bandar signifikan.
            - Validasi breakout/breakdown.
            
            **🎯 Strategi Confluence (Pertemuan Sinyal)**
            - **STRONG BUY**: Harga cross up VWAP + RVOL > 1.2 + Sentinel Positif.
            - **SELL**: Harga jebol VWAP ke bawah + RSI Overbought menukik.
            """)

    except Exception as e:
        st.error(f"Analysis Failed: {str(e)}")

if __name__ == "__main__":
    main()
