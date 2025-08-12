from tools.context_presence_judge import context_presence_tool




# Build the tool
tool = context_presence_tool()

# Example test input
test_message = "What’s the capital?"

# Call the tool
result = tool.func(test_message)

print("Tool Output:")
print(result)