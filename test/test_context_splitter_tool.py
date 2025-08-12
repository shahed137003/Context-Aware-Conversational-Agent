from tools.context_splitter import context_splitter_tool


tool = context_splitter_tool()

# Sample input
test_input = (
    "Last month, my company transferred me to our new branch in Berlin. "
    "I've been exploring the city and trying local foods. "
    "What are some must-see landmarks? "
    "Also, can you suggest any traditional German dishes I should try?"
)

# Run the tool
result = tool.func(test_input)
print(result)