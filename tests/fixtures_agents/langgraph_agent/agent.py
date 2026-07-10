from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
agent = create_react_agent(llm, tools=[search, book])

def run():
    while True:
        result = agent.invoke({"messages": state})
        if result.done:
            break
