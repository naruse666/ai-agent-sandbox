from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


def main() -> None:
    # model
    llm = ChatOpenAI(model="gpt-5", temperature=0)

    # prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "あなたは全てを3行で要約するプロフェッショナルです。"),
            ("human", "{prompt}"),
        ]
    )

    messages = prompt.format_messages(
        prompt="prompt templateはプロンプト文字列と変数を分離できます。モデルはOpenAI gpt-5を利用しています。"
    )

    # invoke
    resp = llm.invoke(messages)
    print(resp.content)


if __name__ == "__main__":
    main()
