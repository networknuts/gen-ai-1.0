import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Hello, I am aryan"
tokens = enc.encode(text)

print("Tokens:", tokens)

tokens = [24912, 575, 939, 646, 10134]
decoded = enc.decode(tokens)

print(f"Decoded Text: {decoded}")
