import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import concurrent.futures
import stocksense_engine as engine

# UI Modules
import ui.theme
import ui.components
import ui.table

# ============================================================================
# 1. PAGE CONFIG & THEME INJECTION
# ============================================================================
st.set_page_config(
    page_title="StockSense Pro | Institutional Tracker",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Global CSS (Glassmorphism, Fonts, Gradients)
ui.theme.inject_global_css()

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

# SECTOR MAPPING (Expanded Broad Universe: Big, Mid, Small Caps)
SECTOR_MAP = {
    "Financials": ["BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "BRIS.JK", "BBTN.JK", "PNBN.JK", "BTPS.JK", "ARTO.JK", "BNGA.JK", "NISP.JK", "BJBR.JK", "BJTM.JK", "BFIN.JK", "TUGU.JK", "ADMF.JK", "AMAR.JK", "BBYB.JK", "BCIC.JK", "BNLI.JK", "PNBS.JK", "AGRO.JK", "MAYA.JK"],
    "Energy": ["ADRO.JK", "PTBA.JK", "PGAS.JK", "MEDC.JK", "AKRA.JK", "ITMG.JK", "HRUM.JK", "BUMI.JK", "INDY.JK", "ELSA.JK", "DEWA.JK", "DOID.JK", "ENRG.JK", "ABMM.JK", "TOBA.JK", "RAJA.JK", "KKGI.JK", "MBSS.JK", "PSI.JK", "SGER.JK", "IATA.JK", "WINE.JK", "GTSI.JK"],
    "Basic Materials": ["MDKA.JK", "ANTM.JK", "INCO.JK", "TINS.JK", "MBMA.JK", "NCKL.JK", "INTP.JK", "SMGR.JK", "BRPT.JK", "TPIA.JK", "ESSA.JK", "MDKI.JK", "IFSH.JK", "KRAS.JK", "LTLS.JK", "ZINC.JK", "DKFT.JK", "NIKL.JK", "TYRE.JK", "BRMS.JK", "UNNU.JK", "NICL.JK"],
    "Consumer Non-Cyclicals": ["ICBP.JK", "INDF.JK", "MYOR.JK", "KLBF.JK", "UNVR.JK", "CPIN.JK", "JPFA.JK", "HMSP.JK", "GGRM.JK", "CMRY.JK", "SIDO.JK", "AMRT.JK", "MIDI.JK", "ROTI.JK", "STTP.JK", "CLEO.JK", "ULTJ.JK", "GOOD.JK", "WOOD.JK", "AISA.JK"],
    "Telecommunications": ["TLKM.JK", "ISAT.JK", "EXCL.JK", "MTEL.JK", "FREN.JK", "TBIG.JK", "TOWR.JK", "CENT.JK", "SUPR.JK", "LINK.JK", "GHON.JK"],
    "Technology": ["GOTO.JK", "EMTK.JK", "BUKA.JK", "BELI.JK", "WIRG.JK", "MTDL.JK", "MLPT.JK", "DMMX.JK", "GLVA.JK", "KIOS.JK", "UVCR.JK", "DIVA.JK", "NFCX.JK"],
    "Infrastructure": ["JSMR.JK", "WIKA.JK", "PTPP.JK", "ADHI.JK", "META.JK", "CMNP.JK", "IPCC.JK", "IPC.JK", "WEGE.JK", "TOTL.JK", "NRCA.JK", "ACST.JK", "IDPR.JK", "POWR.JK", "KEEN.JK"],
    "Healthcare": ["KLBF.JK", "MIKA.JK", "HEAL.JK", "SILO.JK", "SIDO.JK", "SAME.JK", "RDTX.JK", "PRDA.JK", "TSPC.JK", "KAEF.JK", "IRRA.JK", "PEHA.JK", "BMHS.JK"],
    "Properties": ["CTRA.JK", "BSDE.JK", "PWON.JK", "SMRA.JK", "ASRI.JK", "LPKR.JK", "DMAS.JK", "KIJA.JK", "BEST.JK", "APLN.JK", "PANI.JK", "DILD.JK", "MKPI.JK", "RALS.JK", "LPCK.JK", "GWSA.JK", "MTLA.JK"],
    "Automotive & Heavy": ["ASII.JK", "UNTR.JK", "HEXA.JK", "AUTO.JK", "DRMA.JK", "IMAS.JK", "SMSM.JK", "GJTL.JK", "MPMX.JK", "ALDO.JK"]
}

# Cached data fetcher to reduce API calls
@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def fetch_cached_data(ticker, period="1y"):
    """Wrapper for cached yfinance data."""
    ingest = engine.DataIngestion(ticker, period)
    return ingest.fetch_data()

# ============================================================================
# 2. HELPER FUNCTIONS & CHARTING
# ============================================================================

def plot_advanced_charts(df, ticker):
    """Generates the 4-panel Smart Money chart."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.5, 0.15, 0.15, 0.2],
        subplot_titles=("", "", "", "")
    )
    
    # 1. Price + VWAP + BB
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price', increasing_line_color=COLORS['accent_green'], decreasing_line_color=COLORS['accent_red']
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=df['vwap'], mode='lines', name='VWAP (20)',
        line=dict(color=COLORS['accent_warning'], width=1.5)
    ), row=1, col=1)

    # SMA 200 (Long Term Trend)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['sma_200'], mode='lines', name='SMA 200',
        line=dict(color='white', width=2)
    ), row=1, col=1)
    
    # BB
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], line=dict(color='gray', width=1, dash='dot'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], line=dict(color='gray', width=1, dash='dot'), fill='tonexty', fillcolor='rgba(119,136,153,0.1)', showlegend=False), row=1, col=1)

    # 2. RSI
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], line=dict(color='#58a6ff', width=1.5), name='RSI'), row=2, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=70, y1=70, line=dict(color="white", dash="dot", width=1), row=2, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=30, y1=30, line=dict(color="white", dash="dot", width=1), row=2, col=1)

    # 3. CMF (Smart Money Accumulation/Distribution)
    cmf_colors = [COLORS['accent_green'] if val >= 0 else COLORS['accent_red'] for val in df['cmf']]
    fig.add_trace(go.Bar(
        x=df.index, y=df['cmf'],
        name='CMF',
        marker_color=cmf_colors
    ), row=3, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=0, y1=0, line=dict(color="white", width=1), row=3, col=1)
    
    # 4. RVOL
    colors = [COLORS['accent_warning'] if v >= 1.5 else '#4b5563' for v in df['rvol']]
    fig.add_trace(go.Bar(x=df.index, y=df['rvol'], name='RVOL', marker_color=colors), row=4, col=1)
    fig.add_shape(type="line", x0=df.index[0], x1=df.index[-1], y0=1.0, y1=1.0, line=dict(color="white", width=1), row=4, col=1)

    fig.update_layout(template='plotly_dark', height=800, margin=dict(l=10, r=10, t=30, b=10), showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig.update_xaxes(showgrid=False, rangeslider_visible=False)
    fig.update_yaxes(showgrid=True, gridcolor='#2C3342')
    
    return fig

# ============================================================================
# 3. MAIN APP LOGIC
# ============================================================================

def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'screener'
    if 'selected_ticker' not in st.session_state:
        st.session_state['selected_ticker'] = None

    # --- SIDEBAR ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3429/3429815.png", width=50) # Placeholder or custom logo
        st.markdown('<h2 style="margin-top:0;">StockSense Pro</h2>', unsafe_allow_html=True)
        
        mode = st.radio("MAIN NAVIGATION", ["🔍 Smart Screener", "📈 Chart Deep Dive"], 
                        index=0 if st.session_state['page'] == 'screener' else 1)
        
        if mode == "🔍 Smart Screener":
            st.session_state['page'] = 'screener'
        else:
            st.session_state['page'] = 'deep_dive'
            
        ui.components.sidebar_status()
        
        st.caption("© 2026 Quant Labs")

    # ------------------------------------------------------------------
    # PAGE 1: SMART SCREENER
    # ------------------------------------------------------------------
    if st.session_state['page'] == 'screener':
        ui.components.hero_section("Smart Money Flow", "Institutional Accumulation & Volume Anomaly Detector")
        
        # Guide
        ui.components.quick_start_cards()
        
        # --- INPUT CARD ---
        ui.components.card_start()
        ui.components.section_header("Scanner Configuration", "⚙️")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            # 1. Sector Selection
            sector_options = ["Manual Input / None"] + list(SECTOR_MAP.keys())
            selected_sector = st.selectbox("Select Target Sector", sector_options, help="Choose one of the 10 major IDX sectors.")
            
            # 2. Manual Input
            user_tickers = st.text_area(
                "Manual Ticker Override (Optional)", 
                height=70,
                placeholder="e.g. BBCA.JK, BBRI.JK (Overrides sector selection)"
            )
        
        with col2:
            st.write("")
            st.write("")
            st.write("") 
            # Modern large button
            scan_btn = st.button("INITIATE SCAN", type="primary", use_container_width=True)
        ui.components.card_end()

        # SCAN LOGIC
        if scan_btn:
            scan_mode = "MANUAL"
            final_ticker_list = []
            
            # Decision Engine
            if user_tickers.strip():
                scan_mode = "MANUAL"
                raw_list = user_tickers.upper().replace(" ", "").split(',')
                final_ticker_list = [t for t in raw_list if t]
            elif selected_sector != "Manual Input / None":
                scan_mode = "SECTOR_TOP10"
                final_ticker_list = SECTOR_MAP[selected_sector]
            else:
                st.warning("⚠️ Please select a Sector OR input tickers manually.")
                st.stop()
            
            # --- EXECUTION ---
            st.markdown("### 🔄 AI Engine Running...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            valid_results = []
            cleaned_tickers = list(set(final_ticker_list))
            total_tickers = len(cleaned_tickers)
            
            # Using ThreadPool
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                ingest = engine.DataIngestion() 
                analyzer = engine.SmartMoneyAnalyzer()
                
                future_to_ticker = {executor.submit(ingest.fetch_data, t): t for t in cleaned_tickers}
                completed_count = 0
                
                for future in concurrent.futures.as_completed(future_to_ticker):
                    t = future_to_ticker[future]
                    completed_count += 1
                    
                    progress = int((completed_count / total_tickers) * 100)
                    progress_bar.progress(progress)
                    status_text.text(f"Analyzing {t} ({completed_count}/{total_tickers})")
                    
                    try:
                        df = future.result()
                        is_safe, reason = ingest.check_safety_criteria(df)
                        
                        if is_safe:
                            df = ingest.calculate_indicators(df)
                            latest = df.iloc[-1].to_dict()
                            recent_14d = df.iloc[-14:]
                            latest['max_rvol_14d'] = recent_14d['rvol'].max() if not recent_14d.empty else 0
                            
                            analysis = analyzer.analyze_single_row(pd.Series(latest))
                            
                            valid_results.append({
                                "Ticker": t,
                                "Close": latest['close'],
                                "RVOL (Today)": latest['rvol'],
                                "Max RVOL (14D)": recent_14d['rvol'].max(),
                                "MFI": latest['mfi'],
                                "CMF": latest['cmf'],
                                "Status": analysis['status'],
                                "Signal Score": analysis['score']
                            })
                            
                    except Exception as e:
                        continue

            progress_bar.empty()
            status_text.empty()
            
            if not valid_results:
                st.warning("No stocks passed the Safety Filters (Price > 60, Liquidity > 2B). Try another sector.")
            else:
                df_res = pd.DataFrame(valid_results)
                
                # Rank
                if scan_mode == "SECTOR_TOP10":
                    df_res = df_res.sort_values(by=["Signal Score", "RVOL (Today)"], ascending=[False, False])
                    df_res = df_res.head(10)
                    st.success(f"✅ Displaying Top 10 Opportunities in {selected_sector}")
                else:
                    df_res = df_res.sort_values(by="Max RVOL (14D)", ascending=False)
                    
                st.session_state['scan_results'] = df_res
        
        # RESULTS DISPLAY
        if 'scan_results' in st.session_state:
            res_df = st.session_state['scan_results']
            
            ui.components.section_header("Scanning Results", "📊")
            
            # Use new minimal table component
            event = ui.table.render_results_table(res_df)
            
            # Selection Handling
            if len(event.selection['rows']) > 0:
                idx = event.selection['rows'][0]
                selected_ticker = res_df.iloc[idx]['Ticker']
                st.session_state['selected_ticker'] = selected_ticker
                st.session_state['page'] = 'deep_dive'
                st.rerun()

            st.caption("💡 Select a row to proceed to Deep Dive Analysis.")

    # ------------------------------------------------------------------
    # PAGE 2: DEEP DIVE
    # ------------------------------------------------------------------
    elif st.session_state['page'] == 'deep_dive':
        ticker = st.session_state['selected_ticker']
        if not ticker:
            st.warning("No ticker selected.")
            if st.button("⬅️ Back to Screener"):
                st.session_state['page'] = 'screener'
                st.rerun()
            return

        # Header with Back Button
        st.markdown(f"""
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px;">
             <div>
                <h1 class="tx-h1">{ticker}</h1>
                <p class="tx-sub">Institutional Footprint Analysis</p>
             </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⬅️ Return to Scanner"):
            st.session_state['page'] = 'screener'
            st.rerun()

        # Fetch Data
        ingest = engine.DataIngestion(ticker, period="1y")
        df = ingest.fetch_data()
        
        if df.empty:
            st.error(f"Failed to load data for {ticker}")
            return
            
        df = ingest.calculate_indicators(df)
        last_row = df.iloc[-1]
        analyzer = engine.SmartMoneyAnalyzer()
        analysis = analyzer.analyze_single_row(last_row)
        
        # Prepare Metrics for Grid
        metrics_data = [
            {
                "label": "Current Price",
                "value": f"{last_row['close']:,.0f}",
                "delta": f"{last_row['close'] - df.iloc[-2]['close']:+,.0f}",
                "delta_color": "pos" if last_row['close'] >= df.iloc[-2]['close'] else "neg"
            },
            {
                "label": "Relative Volume",
                "value": f"{last_row['rvol']:.2f}x",
                "delta": "Spike Interest" if last_row['rvol'] > 1.5 else "Normal Flow",
                "delta_color": "pos" if last_row['rvol'] > 1.5 else "neu"
            },
            {
                "label": "Net Money Flow (CMF)",
                "value": f"{last_row['cmf']:.2f}",
                "delta": "Accumulating" if last_row['cmf'] > 0 else "Distributing",
                "delta_color": "pos" if last_row['cmf'] > 0 else "neg"
            },
            {
                "label": "Strength Score",
                "value": f"{analysis['score']}/5",
                "delta": analysis['status'],
                "delta_color": "blue"
            }
        ]
        
        ui.components.metric_card_grid(metrics_data)

        # Main Chart
        ui.components.card_start()
        st.markdown("### 📉 Technical Structure")
        st.plotly_chart(plot_advanced_charts(df, ticker), use_container_width=True)
        ui.components.card_end()
        
        # Narrative Box
        ui.components.card_start()
        st.markdown(f"""
        <div style="border-left: 4px solid {'#34D399' if analysis['score'] > 0 else '#F87171'}; padding-left: 16px;">
            <h3 style="margin-top:0;">🤖 AI Insight</h3>
            <p style="font-size: 1.1rem; line-height: 1.6;">{analysis['details']}</p>
        </div>
        """, unsafe_allow_html=True)
        ui.components.card_end()
        
        # Disclaimer
        st.divider()
        st.caption("⚠️ **Disclaimer**: Quantitative analysis tool. Not financial advice.")

if __name__ == "__main__":
    main()
