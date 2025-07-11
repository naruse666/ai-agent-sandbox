from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from datetime import datetime

import tools.search as search
import tools.write_file as write_file


def create_agent():
    try:
        tools = [search.search, write_file.write_file, search.get_page]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

        agent = create_tool_calling_agent(llm, tools, prompt)

        return AgentExecutor(agent=agent, tools=tools, verbose=True)
    except Exception as e:
        print(f"Error creating agent: {e}")
        raise


def main():
    # now date
    now = datetime.now().strftime('%Y-%m')
    web_search_agent = create_agent()
    web_search_agent.invoke({
        "input": f"""
        Search for the latest news in Japan with Japanese, focusing on significant events.
        Pick the most impactful news story, retrieve its content, summarize in Japanese it.
        Finally, write the summarized content to a file.

        # Instructions
        1. Today's date is {now}.
        """
    })


if __name__ == "__main__":
    main()
