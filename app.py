# app.py — Чатбот-асистент фінансового обліку (Gemini + LangGraph)

import json
import uuid
from datetime import datetime

import pandas as pd
import altair as alt

import streamlit as st
from google import genai
from google.genai import types

from langchain_core.messages import HumanMessage, SystemMessage

# Імпортуємо агента
from agent import (
    create_agent,
    extract_response_text,
    extract_tools_debug,
    MODEL_NAME,
)

# ============================================================
# НАЛАШТУВАННЯ СТОРІНКИ
# ============================================================
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
        font-family: 'Comfortaa', sans-serif; 
    }
    
    /* Зміна фону самого вікна, де відображається чат */
    .stChatMessage {
        background-color: #e6e6fa !important; /*  фон для блоків повідомлень */
        border-radius: 12px;
        margin-bottom: 10px;
    }
    
    /* повідомлення асистента */
    [data-testid="stChatMessageAssistant"] {
        background-color: #e8f1f5 !important; /* Ніжно-блакитний для бота */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# ІНІЦІАЛІЗАЦІЯ
# ============================================================
@st.cache_resource
def get_gemini_client(api_key: str):
    return genai.Client(api_key=api_key)

@st.cache_resource
def get_langgraph_agent(api_key: str, model_name: str):
    return create_agent(api_key, model_name)

api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("❌ Не знайдено GOOGLE_API_KEY у secrets.toml")
    st.stop()

agent = get_langgraph_agent(api_key, MODEL_NAME)

# ============================================================
# СТАН STREAMLIT
# ============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = "Ти корисний фінансовий асистент. Відповідай українською мовою."

# Отримуємо ПЛИННИЙ СТАН графа LangGraph (для лімітів та витрат)
config = {"configurable": {"thread_id": st.session_state.thread_id}}
graph_state = agent.get_state(config)

# Витягуємо дані зі стану графа (якщо вони там вже є)
expenses = graph_state.values.get("expenses", []) if graph_state.values else []
monthly_limit = graph_state.values.get("monthly_limit", 10000.0) if graph_state.values else 10000.0
total_spent = sum(float(e.get("amount", 0)) for e in expenses)

# ============================================================
# UI: ЗАГОЛОВОК
# ============================================================
st.title("💸 AI-Асистент Особистих Фінансів")
st.write("Керуйте бюджетом голосом чи текстом: додавайте витрати, встановлюйте ліміти та запитуйте аналітику.")

# ============================================================
# UI: БІЧНА ПАНЕЛЬ (МОНІТОРИНГ БЮДЖЕТУ)
# ============================================================
with st.sidebar:
    st.header("📊 Мій Бюджет")
    
    # 1. Віджет ліміту та прогресу
    st.metric(label="Поточний ліміт", value=f"{monthly_limit:.2f} грн")
    st.metric(label="Усього витрачено", value=f"{total_spent:.2f} грн")

    # --- ЗАЛИШОК ---
    remaining_budget = monthly_limit - total_spent
    
    # Якщо залишок позитивний — показуємо його зеленим, якщо негативний (овердрафт) — червоним
    if remaining_budget >= 0:
        delta_text = f"Доступно: {remaining_budget:.2f} грн"
        delta_color = "normal"  # Зелений колір (позитивний баланс)
    else:
        delta_text = f"Перевищення на: {abs(remaining_budget):.2f} грн"
        delta_color = "inverse" # Червоний колір (перевищення ліміту)
        
    st.metric(
        label="Залишок", 
        value=f"{remaining_budget:.2f} грн",
        delta=delta_text,
        delta_color=delta_color
    )

    # --- НАЙБІЛЬША КАТЕГОРІЯ ---
    top_category_name = "Немає"
    top_category_value = 0.0

    if expenses:
        # Створюємо словник для підрахунку суми по кожній категорії
        cat_totals = {}
        for e in expenses:
            cat = e.get("category", "інше").strip().capitalize()
            cat_totals[cat] = cat_totals.get(cat, 0.0) + float(e.get("amount", 0))
        
        # Знаходимо категорію з максимальною сумою
        if cat_totals:
            top_category_name = max(cat_totals, key=cat_totals.get)
            top_category_value = cat_totals[top_category_name]

    st.metric(
        label="Найбільша категорія", 
        value=top_category_name,
        delta=f"Витрачено: {top_category_value:.2f} грн" if top_category_value > 0 else "0.00 грн",
        delta_color="off" # Вимикаємо зелений/червоний колір для дельти, робимо її нейтрально сірою
    )
    # ----------------------------------------

    st.divider()
    
    # Розрахунок прогресу для повзунка
    if monthly_limit > 0:
        progress_ratio = min(total_spent / monthly_limit, 1.0)
        st.progress(progress_ratio)
        
        # Попередження про перевищення
        if total_spent >= monthly_limit:
            st.error("🚨 Ліміт бюджету вичерпано чи перевищено!")
        elif total_spent >= monthly_limit * 0.8:
            st.warning("⚠️ Ви наблизились до 80% вашого ліміту!")
    else:
        st.info("Ліміт не встановлено або дорівнює 0.")

    st.divider()
    

    # 2. ТАБЛИЦЯ ТА ДІАГРАМА З ФІЛЬТРАЦІЄЮ
    if expenses:
        st.subheader("📝 Мої транзакції")
        
        # Перетворюємо список витрат у DataFrame
        df_expenses = pd.DataFrame(expenses)
        # Приводимо категорії до красивого вигляду (з великої літери)
        df_expenses["category"] = df_expenses["category"].str.strip().str.capitalize()
        
        # Створюємо список унікальних категорій для фільтра + варіант "Усі"
        unique_categories = ["Усі категорії"] + sorted(df_expenses["category"].unique().tolist())
        
        # Віджет вибору категорії
        selected_category = st.selectbox(
            "Фільтр за категорією:",
            options=unique_categories,
            index=0
        )
        
        # Фільтруємо DataFrame відповідно до вибору користувача
        if selected_category == "Усі категорії":
            df_filtered = df_expenses
        else:
            df_filtered = df_expenses[df_expenses["category"] == selected_category]
            
        # Відображаємо відфільтровану таблицю (останні 5 записів)
        st.caption(f"Показано: {selected_category}")
        st.dataframe(
            df_filtered[::-1][:5], 
            column_config={
                "amount": "Сума (грн)",
                "category": "Категорія",
                "description": "Опис"
            },
            use_container_width=True,
            hide_index=True
        )
        
        # 3. Діаграма витрат (теж реагує на фільтр)
        st.subheader("📊 Розподіл")
        
        # Групуємо відфільтровані дані
        df_chart = df_filtered.groupby("category", as_index=False)["amount"].sum()
        
        chart = (
            alt.Chart(df_chart)
            .mark_arc(innerRadius=35, stroke="#fff")
            .encode(
                theta=alt.Theta(field="amount", type="quantitative"),
                color=alt.Color(field="category", type="nominal", legend=alt.Legend(title="Категорії")),
                tooltip=[
                    alt.Tooltip(field="category", title="Категорія"),
                    alt.Tooltip(field="amount", title="Сума (грн)", format=".2f")
                ]
            )
            .properties(height=200)
        )

        st.altair_chart(chart, use_container_width=True)

        # МАЛЕНЬКА ПРИМІТКА ДЛЯ КОРИСТУВАЧА
        st.caption("💡 *Підказка: наведіть курсор на сектор діаграми, щоб побачити точну суму витрат.*")

    else:
        st.info("💡 Тут з'являться ваші витрати, коли ви додасте перший запис.")

    st.divider()
    
    # Налаштування режиму чату
    mode = st.radio(
        "Режим роботи",
        ["🛠️ Агент з інструментами", "💬 Звичайний чат (без БД)"],
        index=0,
        key="mode_radio",
    )

    # Кнопки дій
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Очистити", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())[:8]
            # Скидаємо стан графа (пам'ять) через новий thread_id
            st.rerun()

    with col2:
        if st.session_state.get("messages"):
            export_data = {
                "exported_at": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "thread_id": st.session_state.thread_id,
                "monthly_limit": monthly_limit,
                "expenses": expenses,
                "messages": [str(m) for m in st.session_state.messages],
            }
            st.download_button(
                "📥 Експорт",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"finance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

# ============================================================
# РЕНДЕРІНГ ЧАТУ
# ============================================================
# Відображення історії повідомлень з session_state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Поле введення користувача
if user_input := st.chat_input("Напишіть: 'Додай 250 грн на таксі' або 'Який мій ліміт?'"):
    # Відображаємо повідомлення користувача
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        debug_placeholder = st.empty()

        if "Агент" in mode:
            # --- РОБОТА З LANGGRAPH АГЕНТОМ ---
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Відправляємо репліку користувача в граф
            events = agent.stream(
                {"messages": [HumanMessage(content=user_input)]}, 
                config, 
                stream_mode="values"
            )
            
            final_message = None
            all_messages = []
            
            # Читаємо стрім станів графа
            for event in events:
                if "messages" in event:
                    all_messages = event["messages"]
                    final_message = all_messages[-1]
            
            # Витягуємо текст відповіді моделі
            ai_text = extract_response_text(final_message) if final_message else "Не вдалося отримати відповідь."
            response_placeholder.markdown(ai_text)
            
            # Відображення Debug (інструментів), якщо потрібно
            with st.sidebar:
                show_debug = st.checkbox("Показувати дебаг інструментів", value=False)
            if show_debug:
                debug_info = extract_tools_debug(all_messages)
                if debug_info:
                    debug_placeholder.json(debug_info)
            
            st.session_state.messages.append({"role": "assistant", "content": ai_text})
            
            # Важливо: перезапускаємо сторінку, щоб sidebar миттєво оновив ліміти та прогрес-бар
            st.rerun()

        else:
            # --- ЗВИЧАЙНИЙ РЕЖИМ (ФОЛБЕК БЕЗ ІНСТРУМЕНТІВ) ---
            # залишити прямий стрімінг без бази даних:
            from app import stream_gemini_response # Або реалізація функції нижче
            
            ai_text = ""
            # Для простоти викликаємо звичайне генераційне вікно 
            # Тут можна викликати `stream_gemini_response(user_input, ...)`
            response_placeholder.markdown("Цей режим працює без збереження витрат.")