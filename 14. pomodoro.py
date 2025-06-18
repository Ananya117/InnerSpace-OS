import streamlit as st
import time

def pomodoro_panel():
    st.header("🍅 Pomodoro Focus Timer")
    st.markdown("Stay focused with 25-minute sprints and 5-minute breaks!")

    work_minutes = st.number_input("Work duration (minutes)", min_value=1, max_value=60, value=25)
    break_minutes = st.number_input("Break duration (minutes)", min_value=1, max_value=30, value=5)

    if st.button("Start Work Timer"):
        st.success("Work session started! Stay focused 💪")
        run_timer(work_minutes * 60, "Work")

    if st.button("Start Break Timer"):
        st.info("Break time! Relax 😌")
        run_timer(break_minutes * 60, "Break")

def run_timer(seconds, label):
    start = time.time()
    end = start + seconds
    progress_bar = st.progress(0)
    status_text = st.empty()

    while time.time() < end:
        remaining = int(end - time.time())
        mins, secs = divmod(remaining, 60)
        status_text.markdown(f"**{label} Time Remaining: {mins:02d}:{secs:02d}**")
        progress_bar.progress((seconds - remaining) / seconds)
        time.sleep(1)

    status_text.markdown(f"### ⏰ {label} Timer Done!")
    st.balloons()
