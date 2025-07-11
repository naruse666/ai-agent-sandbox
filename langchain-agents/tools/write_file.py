from pydantic import BaseModel
from langchain.tools import tool


class WriteFileArgs(BaseModel):
    file_path: str
    content: str


@tool(args_schema=WriteFileArgs)
def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file at the specified path.

    Args:
        file_path (str): The path to the file where content will be written.
        content (str): The content to write to the file.

    Returns:
        str: A confirmation message indicating success or failure.
    """
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return f"Content successfully written to {file_path}."
    except Exception as e:
        return f"Error writing to file {file_path}: {e}"
