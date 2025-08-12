from tools.context_relevance_checker import context_relevance_checker_tool
# Create the tool
tool = context_relevance_checker_tool()

# Sample JSON input for testing
test_input = """
{
    "user_question": "What are the health benefits of green tea?",
    "candidate_context": [
        "Green tea is rich in antioxidants and may improve brain function.",
        "The Eiffel Tower is located in Paris.",
        "Studies suggest green tea can help with fat loss.",
        "Mount Everest is the tallest mountain in the world."
    ],
    "threshold": 0.2
}
"""

# Run the tool
result = tool.func(test_input)
print(result)
