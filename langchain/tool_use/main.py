"""
Minimal Tool Use
"""


from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage


# Toolを定義
@tool
def add(a: int, b: int) -> int:
    """2つの整数の和を返す"""
    return a + b


def main():
    llm = ChatOpenAI(model='gpt-5')

    # toolをモデルにバインド
    llm_with_tools = llm.bind_tools([add])

    response = llm_with_tools.invoke(
        [
            SystemMessage(content='あなたは正確性に特化した計算機です。Toolがある場合は積極的に使用してください。'),
            HumanMessage(content='23+19+41 はいくつ？ ')
        ]
    )

    print('=== LLM Response Object ===')
    print(response)

    # この時点で Toolは呼ばれていない
    # response (AIMessage) を出力している
    print('\n=== Tool Calls ===')
    print(response.tool_calls)

    # 手動で実行してみる
    if response.tool_calls:
        call = response.tool_calls[0]
        result = add.invoke(call['args'])
        print('\n=== Tool Result ===')
        print(result)


if __name__ == '__main__':
    main()
