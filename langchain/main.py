from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


def main() -> None:
    llm = ChatOpenAI(model="gpt-5", temperature=0)

    resp = llm.invoke([HumanMessage(content="Hello!")])
    print(resp.content)


if __name__ == "__main__":
    main()
