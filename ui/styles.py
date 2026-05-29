# ui/styles.py - Стилі та конфігурація 

import streamlit as st

def apply_page_config ():
    st.set_page_config(
    page_title="Фінансовий AI-Асистент",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded",
)