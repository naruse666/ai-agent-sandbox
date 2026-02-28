"""
Visualize agent graph

pip install grandalf
"""


from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """2つの整数を足し算する"""
    return a + b


def main() -> None:
    model = ChatOpenAI(model='gpt-5', temperature=0)

    agent = create_agent(
        model=model,
        tools=[add],
        system_prompt="計算が必要なら必ず add ツールを使う。"
    )

    # agent -> graph
    if not hasattr(agent, 'get_graph'):
        raise RuntimeError("Not Found for agent.get_graph()")

    graph = agent.get_graph()

    # print('=== graph object ===')
    # print(type(graph), graph)  # langchain_core.runnables.graph.Graph

    # 1 ASCII
    if hasattr(graph, 'print_ascii'):
        print("\n=== ASCII graph ===")
        graph.print_ascii()

    # 2 Mermaid
    if hasattr(graph, 'draw_mermaid'):
        try:
            mermaid = getattr(graph, 'draw_mermaid')()
            print("\n=== Mermaid ===")
            print(mermaid)
        except Exception as e:
            print(f"\n(Mermaid failed: {type(e).__name__}: {e})")

    # 3 それ以外のダンプ (debug)
    for name in ('nodes', 'edges'):
        if hasattr(graph, name):
            try:
                val = getattr(graph, name)
                print(f"\n=== graph.{name} ===")
                print(val)
            except Exception:
                pass


if __name__ == "__main__":
    main()
