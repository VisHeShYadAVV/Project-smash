# modules/session_handler.py

import streamlit as st

def initialize_session_state():
    if "doc_processed" not in st.session_state:
        st.session_state.doc_processed = False
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "last_response" not in st.session_state:
        st.session_state.last_response = None
