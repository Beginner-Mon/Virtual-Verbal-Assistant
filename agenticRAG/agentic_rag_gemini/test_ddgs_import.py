import streamlit as st
import sys

st.write(f"Python: {sys.executable}")

try:
    from ddgs import DDGS
    st.success("ddgs imported OK!")
    ddgs_obj = DDGS()
    results = list(ddgs_obj.text("hello world", max_results=1))
    st.write(f"Search works: {len(results)} results")
except Exception as e:
    st.error(f"ddgs import FAILED: {type(e).__name__}: {e}")

try:
    from duckduckgo_search import DDGS as DDGS2
    st.warning("duckduckgo_search also available (deprecated)")
except Exception as e2:
    st.info(f"duckduckgo_search not available: {e2}")
