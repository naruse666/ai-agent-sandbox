import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from langchain.tools import tool
from itertools import islice
from pydantic import BaseModel


class SearchArgs(BaseModel):
    query: str
    max_results: int = 5


@tool(args_schema=SearchArgs)
def search(query: str, max_results: int = 5) -> str:
    """
    Perform a web search for the given query and return the top result.

    Args:
        query (str): The search query.

    Returns:
        str: The top search result.
    """

    # Search with DuckDuckGo
    res = DDGS().text(query, region="jp-jp", safesearch="off", backend="lite")
    print(f"Search results for '{query}': {res}")

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "body": r.get("body", ""),
        }
        for r in islice(res, max_results)
    ]


class GetPageArgs(BaseModel):
    url: str


@tool(args_schema=GetPageArgs)
def get_page(url: str) -> str:
    """
    Fetch the content of a web page.

    Args:
        url (str): The URL of the web page to fetch.

    Returns:
        str: The content of the web page.
    """

    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract text from the page, stripping out scripts and styles
        for tag in soup(['script', 'style', 'head', 'noscript']):
            tag.decompose()
        plain_text = soup.get_text(separator='\n', strip=True)
        return plain_text
    except requests.RequestException as e:
        return f"Error fetching {url}: {e}"
