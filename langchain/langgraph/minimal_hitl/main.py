"""
LangGraph minimal HITL tool approval loop
"""

from __future__ import annotations
from pathlib import Path
from typing import Annotated, Literal, TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command


# -------- Tool --------
@tool
def write_file(path: str, content: str) -> str:
    """指定パスにテキストを書き込む（デモ用）"""
    p = Path(path)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {p}"


# -------- State --------
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_tool_call: Optional[dict]  # {"id": str, "name": str, "args": dict}


DecisionType = Literal['approve', 'edit', 'reject']


# -------- Nodes --------
def llm_node(state: State) -> dict:
    """LLMがtool callを提案するしないを決める"""
    model = ChatOpenAI(model='gpt-5').bind_tools([write_file])
    ai: AIMessage = model.invoke(state['messages'])
    # tool_call はAIMessage.tool_callに入る
    return {'messages': [ai]}


def route_after_llm(state: State) -> str:
    """tool callがあるなら reviewへ、なければ終了"""
    last = state['messages'][-1]
    tool_calls = getattr(last, 'tool_calls', None)
    if tool_calls:
        return 'review'
    return END


def review_node(state: State) -> dict:
    """
    tool実行前に人間レビューで止める。
    interrupt() のpayloadは __interrupt__ に返る
    再開時、Command(resume=...) が interrupt() の戻り値になる。
    """

    last = state['messages'][-1]
    tool_calls = getattr(last, 'tool_calls', None) or []
    call = tool_calls[0]  # まずは1個だけ扱う
    planned = {'id': call['id'], 'name': call['name'], 'args': call['args']}

    # ここで停止して人間入力をまつ
    decision: dict = interrupt(
        {
            'type': 'tool_approval',
            'planned_action': planned,
            'help': {
                'approval': {'type': 'approve'},
                'edit': {
                    'type': 'edit',
                    'edited_action': {'name': planned["name"], 'args': planned['args']},
                },
                'reject': {'type': 'reject', 'message': '理由(任意)'},
            },
        }
    )

    dtype: DecisionType = decision.get('type', 'reject')

    if dtype == 'approve':
        return {'pending_tool_call': planned}
    if dtype == 'edit':
        edited = decision.get('edited_action') or {}
        return {'pending_tool_call': {
            'id': planned['id'],
            'name': edited.get('name', planned['name']),
            'args': edited.get('args', planned['args']),
        }}
    # reject: LLMに拒否されたフィードバックを渡して再判断
    msg = decision.get('message', 'Tool execution was rejected.')
    feedback = HumanMessage(content=f"[HITL] tool rejected: {msg}")
    return {'messages': [feedback], 'pending_tool_call': None}


def route_after_review(state: State) -> str:
    """pending_tool_call があれば tools, なければ LLMへ戻す."""
    if state.get('pending_tool_call'):
        return 'tools'
    return 'llm'


def tools_node(state: State) -> dict:
    """承認済みtoolを実行して結果をToolMessageとしてmessagesへ追加"""
    call = state.get('pending_tool_call')
    if not call:
        return {}

    call_id = call['id']
    name = call['name']
    args = call['args']

    if name != 'write_file':
        # 最小なので想定外は拒否
        tm = ToolMessage(tool_call_id=call_id,
                         content=f"Unknown tool: {name}")
        return {'messages': [tm], 'pending_tool_call': None}

    result = write_file.invoke(args)
    tm = ToolMessage(tool_call_id=call_id, content=str(result))
    return {'messages': [tm], 'pending_tool_call': None}


# -------- Build graph --------
def build_app():
    g = StateGraph(State)
    g.add_node("llm", llm_node)
    g.add_node('review', review_node)
    g.add_node('tools', tools_node)

    g.add_edge(START, 'llm')
    g.add_conditional_edges('llm', route_after_llm, {
                            'review': 'review', END: END})
    g.add_conditional_edges('review', route_after_review, {
                            'tools': 'tools', 'llm': 'llm'})
    g.add_edge('tools', 'llm')

    return g.compile(checkpointer=MemorySaver())


def main() -> None:
    app = build_app()
    config = {'configurable': {'thread_id': 'lg-hitl-1'}}

    # 1 interruptで止まる
    out = app.invoke(
        {'messages': [HumanMessage(
            content="次の文章を /tmp/demo.txt に保存して。内容: hello langgraph.")]},
        config=config,
    )

    if "__interrupt__" not in out:
        print('No interrupt. Last: ', out['messages'][-1].content)
        return
    print('=== INTERRUPT PAYLOAD ===')
    print(out['__interrupt__'][0].value)

    # 2 再開
    out2 = app.invoke(
        Command(resume={'type': 'approve'}),
        config=config,
    )
    print("\n=== FINAL ===")
    print(out2['messages'][-1].content)


if __name__ == '__main__':
    main()
