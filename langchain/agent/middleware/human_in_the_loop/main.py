"""
Human in the loop (approve)
"""

from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


@tool
def write_file(path: str, content: str) -> str:
    """指定パスにテキストを書き込む（デモ用）"""
    p = Path(path)
    p.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} chars to {p}"


def main() -> None:
    model = ChatOpenAI(model='gpt-5')
    agent = create_agent(
        model=model,
        tools=[write_file],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={
                    # True: approve/edit/reject 全て許可
                    "write_file": True,
                },
                description_prefix="Tool execution pending approval"
            )
        ],
        # HITL はinterruptで止めて再会するため、checkpointが必要
        checkpointer=InMemorySaver(),
        system_prompt="ユーザーが保存を求めたら write_file を使う."
    )

    config = {'configurable': {'thread_id': 'hitl-demo-1'}}

    # 1 まず実行し、interruptで止める
    result = agent.invoke(
        {'messages': [
            {'role': 'user', 'content': '次の文章を /tmp/demo.txt に保存して。内容: hello langchain'}]},
        config=config,
    )

    # 2 interrpt の有無を確認
    if "__interrupt__" not in result:
        print("No interrupt. Last message:")
        print(result['messages'][-1].content)
        return
    print('=== INTERRUPT ===')
    interrupts = result["__interrupt__"]
    print(interrupts)

    # 3 人間がapproveした体で再開
    resumed = agent.invoke(
        Command(resume={'decisions': [{'type': 'approve'}]}),
        config=config,
    )

    print("\n=== RESUMED ===")
    print(resumed['messages'][-1].content)


if __name__ == "__main__":
    main()
