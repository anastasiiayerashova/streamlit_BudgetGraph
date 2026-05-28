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

    /* Зміна фону всього зовнішнього контейнера інпуту */
    [data-testid="stChatInput"] {
        background-color: #e6e6fa !important; 
        border-radius: 14px !important;       
        padding: 4px !important;
    }
    
    /* Повідомлення асистента */
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

    # Налаштування температури для звичайного чату
    temperature = st.slider(
        "Температура (звичайний чат)",
        min_value=0.0, max_value=1.0, value=0.7, step=0.1,
        key="temperature_slider"
    )

    with st.expander("📝 Системний промпт (чат)"):
        sys_prompt_input = st.text_area("Інструкції", value=st.session_state.system_prompt, height=100)
        if st.button("💾 Зберегти", use_container_width=True):
            st.session_state.system_prompt = sys_prompt_input.strip()
            st.toast("Промпт оновлено!")

    st.divider()

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
                "📥 Експорт (JSON)",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"finance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True,
            )

            # --------------------------------------------------------
            # Експорт у Excel через openpyxl
            # --------------------------------------------------------
            if expenses:
                import io
                
                # Конвертуємо список транзакцій у DataFrame та робимо гарні назви колонок
                df_to_excel = pd.DataFrame(expenses)
                
                # Перейменовуємо та впорядковуємо колонки 
                column_mapping = {
                    "amount": "Сума (грн)",
                    "category": "Категорія",
                    "description": "Опис транзакції"
                }
                df_to_excel = df_to_excel.rename(columns=column_mapping)
                
                # Залишаємо лише ті колонки, які є у маппінгу
                available_cols = [col for col in column_mapping.values() if col in df_to_excel.columns]
                df_to_excel = df_to_excel[available_cols]

                # Записуємо у бінарний потік BytesIO за допомогою openpyxl
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    df_to_excel.to_excel(writer, index=False, sheet_name="Витрати")
                
                # Повертаємо покажчик на початок файлу
                buffer.seek(0)

                st.download_button(
                    label="📊 Експорт в Excel",
                    data=buffer,
                    file_name=f"finance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

# ============================================================
# ФУНКЦІЇ ДЛЯ ЗВИЧАЙНОГО РЕЖИМУ (СТРІМІНГ GEMINI)
# ============================================================
def convert_to_gemini_history(messages: list) -> list[types.Content]:
    contents = []
    for msg in messages:
        if msg["role"] not in ["user", "assistant"]:
            continue
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
        )
    return contents

def stream_gemini_response(prompt: str, history: list, system_prompt: str):
    client = get_gemini_client(api_key)
    contents = convert_to_gemini_history(history)
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))

    try:
        stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=2048,
            ),
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"\n\n❌ **Помилка API:** {str(e)}"

# ============================================================
# РЕНДЕРІНГ ТА ОБРОБКА ЧАТУ
# ============================================================
# Відображаємо історію
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Нове повідомлення від користувача
if user_input := st.chat_input("Введіть запит..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # --- РОЗГАЛУЖЕННЯ РЕЖИМІВ ЗАВДЯКИ СЕЛЕКТОРУ ---
        if "Агент" in mode:
            # 🚀 РЕЖИМ №1: LANGGRAPH АГЕНТ
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            events = agent.stream({"messages": [HumanMessage(content=user_input)]}, config, stream_mode="values")
            
            final_message = None
            all_messages = []
            for event in events:
                if "messages" in event:
                    all_messages = event["messages"]
                    final_message = all_messages[-1]
            
            ai_text = extract_response_text(final_message) if final_message else "Помилка відповіді графа."
            response_placeholder.markdown(ai_text)
            st.session_state.messages.append({"role": "assistant", "content": ai_text})
            st.rerun()

        else:
            # 💬 РЕЖИМ №2: ЗВИЧАЙНИЙ ЧАТ (Стрімінг з урахуванням кастомного системного промпту)
            ai_text = ""
            response_stream = stream_gemini_response(
                prompt=user_input, 
                history=st.session_state.messages[:-1], 
                system_prompt=st.session_state.system_prompt  # Передаємо збережений промпт
            )
            
            for chunk in response_stream:
                ai_text += chunk
                response_placeholder.markdown(ai_text + "▌")
            
            response_placeholder.markdown(ai_text)
            st.session_state.messages.append({"role": "assistant", "content": ai_text})

