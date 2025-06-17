import streamlit as st

st.set_page_config(page_title="Innerspace Mood Detector", page_icon="🧠", layout="centered")
from mood_analytics.analytics import mood_analytics_panel
from journal.journal import journal_panel  # importing the function

st.title("🌿 Innerspace OS")
st.write("Welcome back, love! Here's your cozy digital space.")

# Load the journal panel UI
tab1, tab2 = st.tabs(["📝 Journal", "📊 Analytics"])
with tab1:
    journal_panel()
with tab2:
    mood_analytics_panel()

# After journal_panel()
mood_analytics_panel()

