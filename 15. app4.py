# Importing modular panels
import streamlit as st

# Set Streamlit config early
st.set_page_config(
    page_title="Innerspace OS",
    page_icon="🧠",
    layout="wide"
)

from journal.journal import journal_panel
from mood_analytics.analytics import mood_analytics_panel
from pomodoro.pomodoro import pomodoro_panel

# --- CSS for header ---
st.markdown("""
    <style>
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
        }
        .header-title {
            font-size: 2.5rem;
            font-weight: bold;
            margin: 0;
        }
        .tagline {
            font-size: 1.1rem;
            color: #ccc;
            margin: 0.5rem 0 0 0;
        }
        .feature-buttons {
            position: fixed;
            bottom: 30px;
            right: 30px;
            z-index: 9999;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<div class="header-container">', unsafe_allow_html=True)
col_title1, col_title2 = st.columns([1, 6])
with col_title1:
    st.markdown("### 🍅 Pomodoro")
with col_title2:
    st.markdown('<h1 class="header-title">🧠 Innerspace OS</h1>', unsafe_allow_html=True)
    st.markdown('<p class="tagline">Welcome back, love. Your cozy control panel is waiting 💖</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- Main Layout ---
col1, col2 = st.columns([1, 2])
with col1:
    pomodoro_panel()
with col2:
    st.empty()  # Placeholder for panels

# --- Bottom-right Feature Buttons ---
with st.container():
    st.markdown('<div class="feature-buttons">', unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("📝 Journal"):
            st.query_params["journal"] = ""
    with col_btn2:
        if st.button("📊 Mood Stats"):
            st.query_params["mood"] = ""
    st.markdown('</div>', unsafe_allow_html=True)

# --- Conditional Routes ---
query_params = st.query_params
if "journal" in query_params:
    journal_panel()
elif "mood" in query_params:
    mood_analytics_panel()
