import streamlit as st
from journal.journal import journal_panel  # importing the function

st.set_page_config(page_title="Innerspace Mood Detector", page_icon="🧠", layout="centered")

st.title("🌿 Innerspace OS")
st.write("Welcome back, love! Here's your cozy digital space.")

# Load the journal panel UI
journal_panel()
