import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mood_detector.mood import predict_mood

import streamlit as st
from datetime import datetime
import pandas as pd

# File to save journal entries
JOURNAL_FILE = "journal_entries.csv"

def journal_panel():
    st.header("📓 Daily Journal")
    st.markdown("Write your thoughts below. We'll analyze your mood from what you write.")

    entry = st.text_area("What's on your mind today?", height=200)

    if st.button("Submit"):
        if entry.strip():
            mood = predict_mood(entry)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Save entry to CSV
            new_entry = pd.DataFrame([[timestamp, entry, mood]], columns=["timestamp", "entry", "predicted_mood"])
            if os.path.exists(JOURNAL_FILE):
                df = pd.read_csv(JOURNAL_FILE)
                df = pd.concat([df, new_entry], ignore_index=True)
            else:
                df = new_entry
            df.to_csv(JOURNAL_FILE, index=False)

            st.success(f"Entry saved! Your mood seems to be **{mood}** 🧠")
        else:
            st.warning("Please write something before submitting.")

    # Optional: show history
    if os.path.exists(JOURNAL_FILE):
        with st.expander("📜 View Past Entries"):
            history = pd.read_csv(JOURNAL_FILE)
            st.dataframe(history.tail(10), use_container_width=True)
