"""
Step3: LCEL (LangChain Expression Language) で Prompt | LLM | Parser を繋ぐ
"""


from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def main() -> None:
    llm = ChatOpenAI(model='gpt-5', temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', "あなたはシンプルに説明するAIです。"),
            ('human', '{question}')
        ]
    )

    # LCEL: Prompt -> LLM -> String
    chain = prompt | llm | StrOutputParser()

    # chainは dict を受け取れる (テンプレート変数がキーになる)
    result = chain.invoke({'question': 'LCELとはなんですか？'})
    print(result)


if __name__ == '__main__':
    main()
