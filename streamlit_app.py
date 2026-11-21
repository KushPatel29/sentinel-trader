"""Sentinel: gold-only release. Gold signals are the home page."""
import streamlit as st

st.set_page_config(page_title="Sentinel · Gold signals", page_icon="🪙", layout="wide")
page = st.navigation([
    st.Page("gold_app.py", title="Gold signals", icon="🪙", default=True),
    st.Page("gold_bot.py", title="Gold bot lab (GLD)", icon="🤖", url_path="gold-bot"),
])
page.run()
