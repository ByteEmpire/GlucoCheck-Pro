import google.generativeai as genai

# Configure with your API key
genai.configure(api_key='AIzaSyBxGrw0TpXI1a7C4VYj9wUgfMz4D9I4OmQ')  # Replace with your actual key

# List all available models
print("Available Models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name} (Supports generation)")