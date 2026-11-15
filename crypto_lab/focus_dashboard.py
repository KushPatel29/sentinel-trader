from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import streamlit as st
from sentinel_crypto.focus_ui import render_focus

st.set_page_config(page_title="Sentinel | Gold, Oil & Crypto", page_icon="◈", layout="wide")
st.title("Gold, crude oil & crypto")
render_focus()
