from tools.context_presence_judge import context_presence_tool
from langchain.chat_models import ChatOpenAI  # Or from langchain_community.llms import Ollama

llm = ChatOpenAI(model="gpt-3.5-turbo")  
result = context_presence_tool("Explain the attention mechanism.")

print("Tool Output:", result)  