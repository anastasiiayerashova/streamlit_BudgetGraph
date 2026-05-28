# ui/styles.py - Стилі та конфігурація 

import streamlit as st

def apply_page_config ():
    st.set_page_config(
    page_title="Фінансовий AI-Асистент",
    page_icon="💸",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Nunito', monospace !important;
    }
    
    /* Зміна фону самого вікна, де відображається чат */
    .stChatMessage {
        background-color: #e6e6fa !important; /*  фон для блоків повідомлень */
        border-radius: 12px;
        margin-bottom: 10px;
    }

    /* Зміна фону всього зовнішнього контейнера інпуту */
    [data-testid="stChatInput"] {
        background-color: #ede6fa !important; 
        border-radius: 14px !important;       
        padding: 4px !important;
    }

    [data-testid=stSidebar] {
        background-color: #ede6fa;
        font-family: 'Nunito', monospace !important;
    }
    
    /* Повідомлення асистента */
    [data-testid="stChatMessageAssistant"] {
        background-color: #e8f1f5 !important; /* Ніжно-блакитний для бота */
    }
    </style>
    """,
    unsafe_allow_html=True
)