"""Sample Azure OpenAI chatbot to exercise azure_openai detection."""

import os
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-06-01",
)

SYSTEM_PROMPT = "You are an internal support assistant for the FleurCorp helpdesk."


def answer(question: str) -> str:
    """Answer a user question using the configured Azure deployment."""
    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content
