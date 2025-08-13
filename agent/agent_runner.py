from langchain.agents import create_react_agent, AgentExecutor
from tools.Context_Presence_Judge import context_presence_tool
from tools.Web_Search import web_search_tool
from langchain.llms.fake import FakeListLLM
from langchain.prompts import PromptTemplate
from tools.context_relevance_checker import context_relevance_checker_tool
from tools.context_splitter import context_splitter_tool


mock_llm = FakeListLLM(responses=[
  
    "Action: context_presence_tool\nAction Input: What is LangChain used for?",  
    "Final Answer: Context is missing, need to search.",
    

    "Action: context_relevance_checker_tool\nAction Input: LangChain is an LLM framework",  
    "Final Answer: Context is relevant.",

   
    "Action: context_splitter_tool\nAction Input: I recently learned about LangChain. How do I integrate it with OpenAI's API?",  
    "Final Answer: Background separated from question.",

    
    "Action: web_search_tool\nAction Input: Latest LangChain features",  
    "Final Answer: Found relevant LangChain updates."
])

prompt_template = PromptTemplate(
    input_variables=["input", "agent_scratchpad", "tool_names", "tools"],
    template="""
You are a helpful assistant.

You have access to the following tools:
{tools}

When given a question, think step-by-step and decide if you should use a tool.  
Available tool names: {tool_names}

Question: {input}

{agent_scratchpad}
"""
)

# Define tools
tools = [
    web_search_tool(),
    context_presence_tool(),
    context_relevance_checker_tool(),
    context_splitter_tool()
]

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

# TEST CASES to hit all tools
test_inputs = [
    "What is LangChain used for?",  
    "LangChain is an LLM framework — is this relevant to my question about AI tools?",  
    "I recently learned about LangChain. How do I integrate it with OpenAI's API?", 
    "Latest LangChain features"
]

for i, test in enumerate(test_inputs, 1):
    print(f"\n=== Running Test {i} ===")
    result = agent_executor.invoke({"input": test})
    print("Final Result:", result["output"])
