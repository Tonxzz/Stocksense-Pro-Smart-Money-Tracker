import streamlit as st

def hero_section(title, subtitle):
    st.markdown(f"""
    <div style="text-align: center; padding: 4rem 0;">
        <h1 class="tx-h1">{title}</h1>
        <p class="tx-sub">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def card_start():
    st.markdown('<div class="tx-card">', unsafe_allow_html=True)

def card_end():
    st.markdown('</div>', unsafe_allow_html=True)

def section_header(title, icon="⚡"):
    st.markdown(f"""
    <div class="tx-h2">{icon} {title}</div>
    """, unsafe_allow_html=True)

def quick_start_cards():
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px;">
        <div class="tx-card" style="margin:0;">
            <div style="font-size: 2rem; margin-bottom: 10px;">🎯</div>
            <h3 style="color:white; margin:0 0 5px 0;">1. Select Sector</h3>
            <p style="color:#94A3B8; font-size: 0.9rem;">Choose a sector to scan heavily liquid stocks automatically.</p>
        </div>
        <div class="tx-card" style="margin:0;">
            <div style="font-size: 2rem; margin-bottom: 10px;">⚡</div>
            <h3 style="color:white; margin:0 0 5px 0;">2. Scan Market</h3>
            <p style="color:#94A3B8; font-size: 0.9rem;">Click the big red button to trigger the Smart Money AI Engine.</p>
        </div>
        <div class="tx-card" style="margin:0;">
            <div style="font-size: 2rem; margin-bottom: 10px;">📊</div>
            <h3 style="color:white; margin:0 0 5px 0;">3. Analyze Flow</h3>
            <p style="color:#94A3B8; font-size: 0.9rem;">Review Top 10 results and deep dive into institutional footprints.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def sidebar_status(status="ONLINE", version="2.1.0", updated="Just Now", source="Yahoo Finance"):
    st.sidebar.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); border-radius: 12px; padding: 16px; margin-top: 20px;">
        <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 4px;">SYSTEM STATUS</div>
        <div style="color: #34D399; font-weight: 600; display: flex; align-items: center; gap: 6px;">
            <span style="width: 8px; height: 8px; background: #34D399; border-radius: 50%; box-shadow: 0 0 10px #34D399;"></span>
            {status}
        </div>
        <div style="height: 1px; background: rgba(255,255,255,0.1); margin: 12px 0;"></div>
        <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 2px;">ENGINE VERSION</div>
        <div style="color: #E2E8F0; font-family: monospace;">v{version}</div>
        <div style="margin-top: 8px; font-size: 0.8rem; color: #94A3B8; margin-bottom: 2px;">DATA SOURCE</div>
        <div style="color: #E2E8F0;">{source}</div>
    </div>
    """, unsafe_allow_html=True)

def metric_card_grid(metrics):
    # metrics is list of dict: {label, value, delta, delta_color}
    html_content = '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 24px;">'
    
    for m in metrics:
        color_class = "bg-gray"
        if m.get('delta_color') == 'pos': color_class = "bg-green"
        elif m.get('delta_color') == 'neg': color_class = "bg-red"
        elif m.get('delta_color') == 'blue': color_class = "bg-blue"
        
        delta_html = f'<span class="tx-badge {color_class}">{m.get("delta")}</span>' if m.get('delta') else ''
        
        html_content += f"""
        <div class="tx-card" style="margin:0; padding: 20px;">
            <div style="font-size: 0.85rem; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">{m['label']}</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: white; margin-bottom: 8px;">{m['value']}</div>
            {delta_html}
        </div>
        """
    html_content += "</div>"
    st.markdown(html_content, unsafe_allow_html=True)
