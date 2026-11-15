"""Public release entrypoint; focused market research is the default page."""
import streamlit as st

page=st.navigation([
    st.Page("crypto_lab/release_home.py",title="Live markets",icon="📡",default=True),
    st.Page("crypto_lab/focus_dashboard.py",title="Gold & oil research",icon="🧪",url_path="commodity-research"),
    st.Page("crypto_lab/dashboard.py",title="Archived evidence & paper",icon="📊",url_path="evidence"),
    st.Page("legacy_analyzer_app.py",title="Stock & asset analyzer",icon="🔎",url_path="asset-analyzer")
])
page.run()
