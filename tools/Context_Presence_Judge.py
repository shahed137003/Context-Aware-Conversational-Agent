## this too to decide whether the user provide enough information 

from langchain.tools import Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def build_context_presence_tool(llm):
    # Load the prompt template from a text file
    with open("prompts/context_judge_prompt.txt", "r") as f:
        prompt_template = f.read()

    # Create a LangChain prompt
    prompt = PromptTemplate.from_template(prompt_template)

    # Create an LLMChain: Prompt + Model
    chain = LLMChain(llm=llm, prompt=prompt)

    # Wrap the chain into a LangChain Tool
    return Tool.from_function(
        func=chain.run,
        name="ContextPresenceJudge",
        description="Checks if context is present in user input"
    )
