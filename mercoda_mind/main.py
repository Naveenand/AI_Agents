from langchain_core.messages import HumanMessage
from agents.data_agent import data_agent

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-1"}}   # one id = one conversation

    response = data_agent.invoke(
        {"messages": [HumanMessage(content="What are the top five states up sales")]},
        config,
    )

    print(f"Route: {response['route_response']}")
    print(f"Answer: {response['messages'][-1].content}")