from langchain.tools import Tool  # Use this import for LangChain Tool
from wikipedia import summary, exceptions  # Better to import at the top

def search_wikipedia(query: str) -> str:
    """Search Wikipedia for a summary of the query."""
    try:
        return summary(query)
    except exceptions.DisambiguationError as e:
        return f"Your query is ambiguous. Suggestions: {e.options[:3]}"
    except exceptions.PageError:
        return "Page not found on Wikipedia."
    except Exception:
        return "An error occurred during Wikipedia search."

def web_search_tool():
    return Tool.from_function(
        func=search_wikipedia,
        name="WebSearchTool",
        description="Searches Wikipedia if context is missing."
    )

    