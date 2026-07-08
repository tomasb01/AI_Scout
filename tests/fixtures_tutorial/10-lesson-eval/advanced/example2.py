import anthropic

client = anthropic.Anthropic()
# advanced eval demo
msg = client.messages.create(model='claude-sonnet-4-5', max_tokens=100, messages=[{'role': 'user', 'content': 'hi'}])
