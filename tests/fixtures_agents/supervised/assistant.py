from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def run_agent(task):
    result = agent.invoke(task)
    approval = input("Approve this action? (y/n): ")
    if approval == "y":
        execute(result)
