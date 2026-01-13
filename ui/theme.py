import streamlit as st

def inject_global_css():
    st.markdown("""
    <style>
        /* IMPORT FONTS */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
        
        /* GLOBAL RESET */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #EAF0FF;
        }

        /* BACKGROUND */
        .stApp {
            background-color: #0B1020;
            background-image: 
                radial-gradient(at 0% 0%, rgba(46, 168, 255, 0.15) 0px, transparent 50%),
                radial-gradient(at 100% 0%, rgba(255, 75, 75, 0.1) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(30, 41, 59, 0.5) 0px, transparent 50%);
            background-attachment: fixed;
            background-size: cover;
        }

        /* CONTAINER MAX WIDTH */
        .block-container {
            max-width: 1280px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }

        /* CARD STYLE (GLASSMORPHISM) */
        .tx-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            margin-bottom: 24px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .tx-card:hover {
            border-color: rgba(255, 255, 255, 0.15);
        }

        /* TYPOGRAPHY */
        .tx-h1 {
            font-size: 3rem;
            font-weight: 800;
            background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
        }
        .tx-h2 {
            font-size: 1.8rem;
            font-weight: 700;
            color: #FFFFFF;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .tx-sub {
            font-size: 1.1rem;
            color: #94A3B8;
            font-weight: 400;
            margin-bottom: 2rem;
            line-height: 1.6;
        }

        /* BADGES */
        .tx-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .bg-green { background: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .bg-red { background: rgba(239, 68, 68, 0.2); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }
        .bg-blue { background: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid rgba(59, 130, 246, 0.3); }
        .bg-gray { background: rgba(148, 163, 184, 0.2); color: #CBD5E1; border: 1px solid rgba(148, 163, 184, 0.3); }

        /* SIDEBAR STYLING */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] h1 {
            font-size: 1.5rem !important;
            background: linear-gradient(90deg, #2EA8FF, #3B82F6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* CUSTOM BUTTONS */
        .stButton button {
            background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
            color: white;
            font-weight: 600;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            border: none;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
            transition: all 0.3s ease;
            width: 100%;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(239, 68, 68, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        .stButton button:active {
            transform: scale(0.98);
        }

        /* INPUT FIELDS */
        .stTextInput input, .stTextArea textarea, .stSelectbox div[data-baseweb="select"] {
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            color: #F8FAFC;
        }
        .stTextInput input:focus, .stTextArea textarea:focus, .stSelectbox div[data-baseweb="select"]:focus-within {
            border-color: #2EA8FF;
            box-shadow: 0 0 0 2px rgba(46, 168, 255, 0.2);
        }

        /* TABLE WRAPPER */
        .tx-table-wrap {
            background: #141B2D;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 16px;
            overflow: hidden;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* PLOTLY FIX */
        .js-plotly-plot .plotly .modebar {
            background-color: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)
