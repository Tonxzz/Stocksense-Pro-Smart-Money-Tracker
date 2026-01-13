import streamlit as st

def render_results_table(df):
    """
    Renders the results dataframe inside a custom styled wrapper.
    """
    st.markdown('<div class="tx-table-wrap">', unsafe_allow_html=True)
    
    event = st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
            "Close": st.column_config.TextColumn("Price (IDR)"),
            "Status": st.column_config.TextColumn(
                "AI Status",
                help="Smart Money Interpretation",
                validate="^(ACCUMULATION|DISTRIBUTION|MARKUP|NEUTRAL)$"
            ),
            "RVOL (Today)": st.column_config.NumberColumn(
                "RVOL",
                format="%.2fx",
                help="Relative Volume (Today vs 20D Avg)"
            ),
            "Max RVOL (14D)": st.column_config.ProgressColumn(
                "Max Spike (14D)",
                help="Highest Relative Volume in last 2 weeks",
                format="%.2fx",
                min_value=0,
                max_value=5,
            ),
            "Signal Score": st.column_config.NumberColumn(
                "AI Score",
                format="%d ⭐",
                help="Proprietary AI Strength Score (0-5)"
            ),
             "CMF": st.column_config.NumberColumn("Flow (CMF)", format="%.2f"),
             "MFI": st.column_config.NumberColumn("MFI", format="%.1f")
        },
        selection_mode="single-row",
        on_select="rerun"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    return event
