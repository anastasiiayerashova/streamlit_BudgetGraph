# agent.py — LangGraph-агент з фінансовими інструментами (Gemini)

from __future__ import annotations

import os
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict

# LangChain & Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# LangGraph
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver

# ============================================================
# КОНСТАНТИ
# ============================================================
MODEL_NAME = "gemini-2.5-flash"

# ============================================================
# СХЕМА СТАНУ
# ============================================================
def add_expenses(prev: List[Dict], new: List[Dict]) -> List[Dict]:
    return (prev or []) + (new or [])

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]       # Історія повідомлень
    expenses: Annotated[list[dict], add_expenses] # База даних витрат користувача
    monthly_limit: float                          # Ліміт бюджету

# ============================================================
# ІНСТРУМЕНТИ
# ============================================================
@tool
def add_expense(amount: float, category: str, description: str) -> dict:
    """
    Використовуй цей інструмент, щоб ЗАПИСАТИ нову витрату.
    Аргументи: сума (float), категорія (їжа, транспорт, розваги тощо), опис.
    """
    return {"amount": amount, "category": category, "description": description}

@tool
def total_spent() -> str:
    """
    Рахує загальну суму усіх витрат у системі.
    """
    return "trigger_total"

@tool
def summary_by_category() -> dict:
    """
    Використовуй цей інструмент, щоб отримати ПОТОЧНИЙ список усіх витрат та ліміт.
    Завжди викликай його перед підрахунком балансу, пошуком категорій чи перевіркою лімітів.
    """
    return {"status": "request_state_data"}

@tool
def set_limit_tool(new_limit: float) -> str:
    """
    Використовуй цей інструмент, коли користувач просить змінити, оновити або встановити новий місячний ліміт.
    """
    return f"success_limit_{new_limit}"


TOOLS = [add_expense, total_spent, summary_by_category, set_limit_tool]

# ============================================================
# ВУЗЛИ ТА ЛОГІКА ГРАФА
# ============================================================
def custom_tool_node(state: AgentState):
    """Кастомний вузол для виконання інструментів та синхронізації зі State"""
    last_msg = state["messages"][-1]
    new_expenses = []

    updated_limit = state.get("monthly_limit", 10000.0)
    current_expenses = state.get("expenses", [])

    modified_tool_calls = []
    for tool_call in last_msg.tool_calls:
        tc = tool_call.copy()

        # Перехоплюємо зміну ліміту
        if tc["name"] == "set_limit_tool":
            val = tc["args"].get("new_limit") or tc["args"].get("amount")
            if val is not None:
                updated_limit = float(val)

        # Перехоплюємо додавання витрат
        elif tc["name"] == "add_expense":
            new_expenses.append({
                "amount": float(tc["args"].get("amount", 0)),
                "category": str(tc["args"].get("category", "інше")),
                "description": str(tc["args"].get("description", ""))
            })

        modified_tool_calls.append(tc)

    # Запускаємо стандартний виконувач інструментів LangGraph
    standard_tool_node = ToolNode(TOOLS)
    tool_output = standard_tool_node.invoke(state)

    # Модифікуємо контент відповідей для специфічних інструментів
    final_messages = []
    for msg in tool_output["messages"]:
        if isinstance(msg, ToolMessage):
            if msg.name == "summary_by_category":
                msg.content = f"АКТУАЛЬНІ ДАНІ: Ліміт={updated_limit}, Витрати={current_expenses}"
            elif msg.name == "total_spent":
                total = sum(float(e.get("amount", 0)) for e in current_expenses)
                msg.content = f"📊 Загальна сума витрат: {total:.2f} грн. Поточний ліміт: {updated_limit} грн."
        final_messages.append(msg)

    return {
        "messages": final_messages,
        "expenses": new_expenses,
        "monthly_limit": updated_limit
    }


def reporter_node(state: AgentState):
    """Додатковий вузол для генерації красивого звіту"""
    expenses = state.get("expenses", [])
    if not expenses:
        return {"messages": [AIMessage(content="📊 Ваш баланс чистий, витрат немає!")]}

    total = sum(e["amount"] for e in expenses)
    categories = {}
    for e in expenses:
        cat = e["category"].lower()
        categories[cat] = categories.get(cat, 0) + e["amount"]

    report = f"📊 **ГЕНЕРОВАНИЙ АНАЛІТИЧНИЙ ЗВІТ**:\nЗагальна сума: {total:.2f} грн.\n"
    for c, amt in categories.items():
        report += f"- {c.capitalize()}: {amt:.2f} грн ({ (amt/total)*100 :.1f}%)\n"

    return {"messages": [AIMessage(content=report)]}


def router(state: AgentState):
    """Роутер для визначення наступного кроку"""
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    
    # Перевірка на запит звіту
    if len(state["messages"]) > 0 and isinstance(state["messages"][-1], HumanMessage):
        user_text = state["messages"][-1].content.lower()
        if "звіт" in user_text or "аналітика" in user_text:
            return "reporter"
            
    return END

# ============================================================
# ГОЛОВНА ФУНКЦІЯ СТВОРЕННЯ АГЕНТА
# ============================================================
def create_agent(api_key: str, model_name: str = MODEL_NAME):
    """
    Створює скомпільований LangGraph-агент фінансового обліку.
    """
    # Ініціалізуємо модель за допомогою ключа, отриманого зі Streamlit secrets
    llm = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=0.0,
        api_key=api_key
    )
    llm_with_tools = llm.bind_tools(TOOLS)

    def agent_node(state: AgentState):
        system_prompt = SystemMessage(content=(
            f"Ти фінансовий асистент. Поточний ліміт: {state.get('monthly_limit', 0)} грн.\n"
            f"Кількість записів у базі витрат: {len(state.get('expenses', []))}.\n"
            "ЯКЩО тобі потрібно порахувати суму, знайти витрати або перевірити ліміт — ОБОВ'ЯЗКОВО спочатку "
            "викличи інструмент `summary_by_category`, щоб отримати актуальний масив даних для аналізу."
        ))

        messages = [system_prompt] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Будуємо граф
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", custom_tool_node)
    builder.add_node("reporter", reporter_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "reporter": "reporter",
            END: END
        }
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("reporter", END)

    # Додаємо пам'ять
    checkpointer = InMemorySaver()
    return builder.compile(checkpointer=checkpointer)

# Допоміжні функції для інтерфейсу (використовуються в app.py)
def extract_response_text(message) -> str:
    """Витягує текст з повідомлення LangChain."""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)

def extract_tools_debug(messages: list[Any]) -> list[dict]:
    """Повертає короткий список викликів інструментів/результатів для debug UI."""
    debug = []
    for m in messages:
        if isinstance(m, ToolMessage):
            debug.append({"type": "tool_result", "content": m.content, "tool_call_id": m.tool_call_id})
        else:
            tool_calls = getattr(m, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    debug.append({
                        "type": "tool_call",
                        "name": tc.get("name"),
                        "args": tc.get("args"),
                        "id": tc.get("id"),
                    })
    return debug