from tools.Web_Search import web_search_tool


tool = web_search_tool()

# Test case 1 – direct query
print("\n--- Test 1: Clear Query ---")
result = tool.func("Albert Einstein")
print(result)

# Test case 2 – ambiguous query
print("\n--- Test 2: Ambiguous Query ---")
result = tool.func("Mercury")  # Could mean planet, element, god
print(result)

# Test case 3 – page not found
print("\n--- Test 3: Page Not Found ---")
result = tool.func("asdkjhqweoiuqwe") 
print(result)