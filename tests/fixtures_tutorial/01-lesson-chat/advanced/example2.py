import openai

client = openai.OpenAI()
# advanced chat demo — same flow, second variant
resp = client.chat.completions.create(model='gpt-4o-mini', messages=[{'role': 'user', 'content': 'hello again'}])
print(resp.choices[0].message.content)
