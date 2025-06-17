import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

JOURNAL_FILE = "journal_entries.csv"

def mood_analytics_panel():
    st.header("📊 Mood Insights")
    
    try:
        df = pd.read_csv(JOURNAL_FILE)
    except FileNotFoundError:
        st.warning("No journal entries found yet.")
        return

    if df.empty:
        st.warning("No journal data to analyze.")
        return

    # Mood Distribution Pie
    st.subheader("🧠 Mood Distribution")
    mood_counts = df['predicted_mood'].value_counts()
    st.bar_chart(mood_counts)

    # Mood Over Time Line Plot
    st.subheader("📈 Mood Over Time")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_sorted = df.sort_values("timestamp")
    df_sorted['mood_numeric'] = df_sorted['predicted_mood'].apply(lambda x: 1 if "Good" in x else 0)

    st.line_chart(df_sorted.set_index("timestamp")['mood_numeric'])

    # Mood Emoji Breakdown
    st.subheader("😸 Emoji Mood Breakdown")
    emoji_counts = df['predicted_mood'].value_counts()
    st.write(emoji_counts)

