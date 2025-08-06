from langchain.agents import create_react_agent, AgentExecutor
from tools.Context_Presence_Judge import context_presence_tool
from tools.Web_Search import web_search_tool
from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate


mock_llm = FakeListLLM(responses=[
    "context_missing", 
    "LangChain is used for building LLM-powered applications."
])


prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    template="""
You are a helpful assistant.

Question: {input}
{agent_scratchpad}
"""
)

# Define tools
tools = [web_search_tool(), context_presence_tool()]

# Create the agent
agent = create_react_agent(
    llm=mock_llm,
    tools=tools,
    prompt=prompt_template
)

# Create executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# Run test
result = agent_executor.invoke({"input": "What is LangChain used for?"})

print("\nFinal Result:", result["output"])
