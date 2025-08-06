from tools.context_presence_judge import build_context_presence_tool
from langchain.chat_models import ChatOpenAI  # Or from langchain_community.llms import Ollama

llm = ChatOpenAI(model="gpt-3.5-turbo")  
tool = build_context_presence_tool(llm)

# Example test
result = tool.run("Explain the attention mechanism.")
print("Tool Output:", result)  