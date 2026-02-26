"""
Minimal Agent
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


# ----------------
# Tool
# ----------------
@tool
def add(a: int, b: int) -> int:
    """2つの整数を足し算する"""
    return a+b


def main():

    # model
    model = ChatOpenAI(
        model='gpt-5',
        temperature=0,
    )

    # agent
    agent = create_agent(
        model=model,
        tools=[add],
        system_prompt="あなたは計算が必要なら必ずツールを使うアシスタントです。"
    )

    # agent実行
    result = agent.invoke(
        {"messages": [
            {"role": "user", "content": "23+11+8は？"}
        ]}
    )
    print(result)


if __name__ == "__main__":
    main()
