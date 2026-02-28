"""
Observe execution via agent.stream()
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """2つの整数を足し算する"""
    return a + b


def main() -> None:
    model = ChatOpenAI(model='gpt-5')
    agent = create_agent(
        model=model,
        tools=[add],
        system_prompt="計算が必要なら必ず add ツールを使う。",
    )

    state_in = {'messages': [{'role': "user", 'content': '23+19 = ?'}]}

    for i, event in enumerate(agent.stream(state_in)):
        print(f"\n--- event {i} ---")
        print(event)


if __name__ == "__main__":
    main()
