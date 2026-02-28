"""
Extend AgentState and pass extra fields
"""

from langchain.agents import create_agent, AgentState
from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime


# AgentState を拡張
class CustomState(AgentState):
    user_name: str
    preferences: dict


@tool
def greet(runtime: ToolRuntime[None, CustomState]) -> str:
    """ユーザー名で挨拶する(stateからuser_nameを読む)"""
    user_name = runtime.state.get("user_name", "Unknown")
    return f"Hello {user_name}"


def main() -> None:
    model = ChatOpenAI(model='gpt-5')
    agent = create_agent(
        model=model,
        tools=[greet],
        state_schema=CustomState,
        system_prompt="挨拶を求められたら greet ツールを使う"
    )

    state_out = agent.invoke(
        {
            "messages": [{"role": "user", "content": "Hello"}],
            "user_name": "Bob",
            "preferences": {"language": "English"},
        }
    )

    print("=== keys ===")
    print(state_out.keys())
    print("user_id:", state_out.get("user_id"))
    print("preferences: ", state_out.get("preferences"))
    print("last: ", state_out["messages"][-1].content)


if __name__ == "__main__":
    main()
