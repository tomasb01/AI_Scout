import anthropic

client = anthropic.Anthropic()
# advanced streaming demo
msg = client.messages.create(model='claude-sonnet-4-5', max_tokens=100, messages=[{'role': 'user', 'content': 'hi'}])
