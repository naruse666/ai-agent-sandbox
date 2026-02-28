"""
Middleware (Model call limit)
"""

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver


@tool
def add(a: int, b: int) -> int:
    """2つの整数を足し算する"""
    return a + b


def main() -> None:
    model = ChatOpenAI(model="gpt-5", temperature=0)
    agent = create_agent(
        model=model,
        tools=[add],
        # ModelCallLimitMiddleware はスレッド管理のため、checkpointerが必要
        checkpointer=InMemorySaver(),
        middleware=[
            ModelCallLimitMiddleware(
                thread_limit=10,
                run_limit=2,  # invoke内のモデル呼び出し上限
                exit_behavior="end",
            )
        ],
        system_prompt="計算は必ず add ツールを使う",
    )

    # tool呼び出しが入ると、通中はモデル呼び出しが複数回になる
    out = agent.invoke(
        {'messages': [{'role': 'user', 'content': '11+21+2+4=?'}]},
        config={'configurable': {'thread_id': 't1'}},
    )

    # print(out['messages'][-1].content)
    print(out['messages'][-1])


if __name__ == "__main__":
    main()
