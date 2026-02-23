"""
Runnable Runtime
"""


import asyncio

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# 共通 Chain


def build_chain():
    llm = ChatOpenAI(model='gpt-5', temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', '簡潔に答えるアシスタントです。'),
            ('human', '{question}')
        ]
    )

    return prompt | llm | StrOutputParser()


# invoke
def example_invoke(chain):
    print("\n=== invoke ===")
    print(chain.invoke({'question': 'LangChainとは？'}))


# batch
def example_batch(chain):
    print('\n=== batch ===')
    inputs = [
        {'question': 'LCELとは？'},
        {'question': 'Runnableとは？'},
        {'question': 'Agentとは？'},
    ]

    results = chain.batch(inputs)
    for r in results:
        print('-', r)


# stream
def example_stream(chain):
    print('\n=== stream ===')

    for chunk in chain.stream({'question': 'LangChainを説明して。'}):
        print(chunk, end="", flush=True)
    print()


# async invoke
async def example_async(chain):
    print('\n=== ainvoke ===')

    result = await chain.ainvoke({'question': 'asyncとは？'})
    print(result)


# retry
def example_retry(chain):
    print('\n=== retry ===')

    retry_chain = chain.with_retry(
        stop_after_attempt=3
    )

    print(retry_chain.invoke({'question': 'retryとは？'}))


# RunnableParallel (並列分岐)
def example_parallel():
    print('\n=== RunnableParallel ===')

    llm = ChatOpenAI(model='gpt-5', temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ('human', "{topic}を一言で説明。")
        ]
    )

    chain = prompt | llm | StrOutputParser

    parallel = RunnableParallel(
        short=chain,
        another=chain,
    )

    result = parallel.invoke({"topic": "LangChain"})
    print(result)


def main():
    chain = build_chain()

    example_invoke(chain)
    example_batch(chain)
    example_stream(chain)
    asyncio.run(example_async(chain))
    example_retry(chain)

    # TypeError: BaseModel.__init__() takes 1 positional argument but 2 were given
    # example_parallel()


if __name__ == '__main__':
    main()
