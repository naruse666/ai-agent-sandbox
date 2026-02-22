"""
PydanticOutputParser で構造化出力 (LCEL)
"""

from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class Answer(BaseModel):
    topic: str = Field(description="対象トピック")
    summary: str = Field(description="1文要約 (日本語)")
    confidence: Literal["low", "medium", "high"] = Field(description="自身度")


def main() -> None:
    llm = ChatOpenAI(model='gpt-5', temperature=0)

    parser = PydanticOutputParser(pydantic_object=Answer)

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', 'あなたは厳密に出力形式を守るアシスタントです。'),
            (
                'human',
                "次のトピックについて1文で要約し、自身度もつけてください。\n"
                "トピック: {topic}\n\n"
                "{format_instructions}",
            )
        ]
    )

    chain = prompt | llm | parser
    result: Answer = chain.invoke(
        {
            'topic': 'LCEL (LangChain Expression Language)',
            'format_instructions': parser.get_format_instructions(),
        }
    )

    # dict として扱える
    print(result.model_dump())


if __name__ == '__main__':
    main()
