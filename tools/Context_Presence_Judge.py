from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

def context_presence_function(message: str) -> str:
    """Check whether the user provided enough context using a local Hugging Face model."""

    # Load the prompt template from file
    with open("prompts/context_judge_prompt.txt", "r") as file:
        template = file.read()

    # Create the prompt object
    prompt_template = ChatPromptTemplate.from_template(template)
    prompt_value = prompt_template.invoke({"input": message})

    # Create a local Hugging Face pipeline
    generator = pipeline(
        model="google/flan-t5-small",  
        task="text2text-generation",
        max_new_tokens=50
    )

    # Wrap in LangChain's HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=generator)

    # Run the model
    result = llm.invoke(prompt_value.to_string())  # Send the full prompt as text

    return result

def context_presence_tool():
    """Builds the LangChain Tool for checking context presence (offline)."""
    return Tool.from_function(
        func=context_presence_function,
        name="ContextPresenceJudge",
        description="Determines whether the user gave enough background/context."
    )

# def build_context_presence_tool(llm):
#     # Load the prompt template from a text file
#     with open("prompts/context_judge_prompt.txt", "r") as f:
#         prompt_template = f.read()

#     # Create a LangChain prompt
#     prompt = PromptTemplate.from_template(prompt_template)

#     # Create an LLMChain: Prompt + Model
#     chain = LLMChain(llm=llm, prompt=prompt)

#     # Wrap the chain into a LangChain Tool
#     return Tool.from_function(
#         func=chain.run,
#         name="ContextPresenceJudge",
#         description="Checks if context is present in user input"
#     )
