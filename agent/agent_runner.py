from langchain.agents import create_react_agent, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline

# ---- Load Hugging Face Model ----
# GPT-2 is not chat-tuned, so outputs might not be ideal; 
# consider 'mistralai/Mistral-7B-Instruct-v0.2' if your hardware allows
generator = pipeline(
    task="text-generation",
    model="gpt2",
    max_new_tokens=256,
    temperature=0.7
)

hf = HuggingFacePipeline(pipeline=generator)

# ---- Load Prompt ----
prompt = hub.pull("hwchase17/react")  # Contains the required input variables

tools = [
    web_search_tool(),
    context_presence_tool(),
    context_relevance_checker_tool(),
    context_splitter_tool()
]

# ---- Memory ----
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ---- Create Agent ----
agent = create_react_agent(
    llm=hf,
    tools=tools,
    prompt=prompt
)

# ---- Create Agent Executor ----
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

# ---- Initial System Context ----
memory.chat_memory.add_ai_message(
    "You are an AI assistant that uses tools to provide helpful answers. "
    "If you can't answer something, say so clearly."
)

# ---- Chat Loop ----
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = agent_executor.invoke({"input": user_input})
    print("ChatBot:", response.get("output", "No response generated."))


