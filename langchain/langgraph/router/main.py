"""
LangGraph Router Example
"""
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated


# --- State ---
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# --- Nodes ---
def router_node(state: State) -> dict:
    return {}


def route_after_router(state: State) -> str:
    """Routing"""
    question = state['messages'][-1].content.lower()

    if any(x in question for x in ["+", "-", "*", "/", "計算"]):
        return 'math'
    return 'chat'


def math_node(state: State):
    question = state['messages'][-1].content
    result = f"[math node] I think this is a math question: {question}"
    return {'messages': [AIMessage(content=result)]}


def chat_node(state: State):
    question = state['messages'][-1].content
    result = f"[chat node] Normal conversation: {question}"
    return {'messages': [AIMessage(content=result)]}


def build_app():
    graph = StateGraph(State)

    graph.add_node('router', router_node)
    graph.add_node('math', math_node)
    graph.add_node('chat', chat_node)

    graph.add_edge(START, 'router')
    graph.add_conditional_edges(
        'router',
        route_after_router,
        {
            "math": 'math',
            "chat": 'chat',
        },
    )

    graph.add_edge('math', END)
    graph.add_edge('chat', END)

    return graph.compile()


# --- Run ---
def main():
    app = build_app()

    result = app.invoke(
        {'messages': [HumanMessage(content="22 + 11 = ?")]}
    )

    print(result['messages'][-1].content)


if __name__ == "__main__":
    main()
