"""
with_structured_output: v1で推奨の方法
"""


from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model='gpt-5', temperature=0)


class Book(BaseModel):
    title: str = Field(description='本のタイトル')
    author: str = Field(description='著者名')
    year: int = Field(description='出版年(西暦)')


llm = model.with_structured_output(Book)
text = "『The Pragmatic Programmer』は Andrew Hunt と David Thomas が1999年に出版。"

out = llm.invoke(text)
print(out)
