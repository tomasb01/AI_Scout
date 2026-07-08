import openai

client = openai.OpenAI()
# functions demo
resp = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role': 'user', 'content': 'hi'}])
print(resp.choices[0].message.content)
