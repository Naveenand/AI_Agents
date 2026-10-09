import os
import sys
import uuid

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from agents.etl_agent import etl_analyst
from agents.sql_agent import sql_analyst
from models.schema import DataAgentSchema, RouterSchema
from utils.llm_pickup import pick_llm

ROUTER_HISTORY = 6        # how many recent messages the router sees
SQL_HISTORY = 6           # how many recent messages are used to rewrite a SQL follow-up

# Describe your data here: the router uses it to decide sql vs etl.
DATABASE_DESCRIPTION = (
    "The Olist Brazilian e-commerce PostgreSQL database: orders, order_items, order_payments, "
    "order_reviews, customers, sellers, products, product_category_name_translation, geolocation. "
    "It covers sales, revenue, payments, deliveries, reviews, customers and sellers by state/city."
)

ROUTER_PROMPT = f"""You route each user message to the right specialist.

- sql: questions answered by querying the company database. {DATABASE_DESCRIPTION}
  Examples: "Top 5 states by sales", "How many orders were delivered late?",
  "Average review score per product category", "Now only for 2018".
- etl: downloading data from an API URL, working with data files (csv/json/parquet),
  filtering or transforming files, and saving or re-running pipelines.
  Examples: "Get data from https://...", "Keep grass pokemon in that file", "Save this as a pipeline",
  "What files do I have?", "Run the grass_strong pipeline".
- clarify: the message is too unclear to choose (e.g. "fix it", "show me the data" with no context).
  Then write one short clarifying question.

Use the conversation: a short follow-up like "save this", "now by city" or "do the same for fire"
belongs to the same specialist as the previous request."""

router_llm = pick_llm("medium").with_structured_output(RouterSchema)
rewrite_llm = pick_llm("low")


# ---------------------------- NODES ---------------------------- #

def router_node(state: DataAgentSchema):
    recent = state.messages[-ROUTER_HISTORY:]
    try:
        decision = router_llm.invoke([SystemMessage(content=ROUTER_PROMPT)] + recent)
        return {"route_response": decision.answer,
                "clarifying_question": decision.clarifying_question}
    except Exception as e:
        # If the router itself fails, ask the user instead of crashing.
        return {"route_response": "clarify",
                "clarifying_question": "Sorry, I couldn't tell what you need. Is this a question "
                                       "about the database, or a task on an API or data file?",
                "last_error": f"router failed: {e}"}


def clarify_node(state: DataAgentSchema):
    question = state.clarifying_question or "Could you give me a bit more detail about what you need?"
    return {"messages": [AIMessage(content=question)]}


def standalone_question(state: DataAgentSchema) -> str:
    """Turn a follow-up like 'now split that by city' into a full question the SQL agent can use."""
    question = state.messages[-1].content
    earlier = state.messages[:-1][-SQL_HISTORY:]
    if not earlier:
        return question
    conversation = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}" for m in earlier)
    prompt = f"""Rewrite the user's latest question so it can be understood on its own,
using the conversation for missing details (which table, metric, filter, time period).
If it already makes sense on its own, return it unchanged. Return only the question.

Conversation:
{conversation}

Latest question: {question}"""
    try:
        rewritten = rewrite_llm.invoke(prompt).content.strip()
        return rewritten or question
    except Exception:
        return question


def sql_node(state: DataAgentSchema):
    question = standalone_question(state)
    try:
        result = sql_analyst.invoke({"user_question": question})
    except Exception as e:
        return {"messages": [AIMessage(content=f"The SQL analyst ran into a problem: {e}")],
                "last_error": str(e), "last_sql": ""}

    answer = result.get("final_answer") or (
        f"I couldn't answer that from the database. {result.get('error_message', '')}".strip())
    return {"messages": [AIMessage(content=answer)],
            "last_sql": result.get("generated_sql_query", ""),
            "last_error": result.get("error_message", "")}


def etl_node(state: DataAgentSchema):
    # Give the ETL agent ITS OWN history (with tool calls and results), so it knows
    # what "this", "that file" or "the grass one" refers to.
    etl_input = state.etl_messages + [HumanMessage(content=state.messages[-1].content)]
    try:
        result = etl_analyst.invoke({"messages": etl_input, "step_count": 0})
    except Exception as e:
        return {"messages": [AIMessage(content=f"The ETL analyst ran into a problem: {e}")],
                "last_error": str(e)}

    return {"messages": [AIMessage(content=result["messages"][-1].content)],
            "etl_messages": result["messages"],
            "last_error": ""}


def route_edge(state: DataAgentSchema) -> str:
    return {"sql": "sql_node", "etl": "etl_node"}.get(state.route_response, "clarify_node")


# ---------------------------- GRAPH ---------------------------- #

data_agent_graph = StateGraph(DataAgentSchema)
data_agent_graph.add_node("router_node", router_node)
data_agent_graph.add_node("sql_node", sql_node)
data_agent_graph.add_node("etl_node", etl_node)
data_agent_graph.add_node("clarify_node", clarify_node)

data_agent_graph.add_edge(START, "router_node")
data_agent_graph.add_conditional_edges("router_node", route_edge,
                                       {"sql_node": "sql_node", "etl_node": "etl_node",
                                        "clarify_node": "clarify_node"})
data_agent_graph.add_edge("sql_node", END)
data_agent_graph.add_edge("etl_node", END)
data_agent_graph.add_edge("clarify_node", END)

# MemorySaver keeps each conversation's state between messages (by thread_id),
# so you only send the NEW message each time.
data_agent = data_agent_graph.compile(checkpointer=MemorySaver())


def ask(question: str, thread_id: str) -> dict:
    """Send one message in a conversation and return the updated state."""
    config = {"configurable": {"thread_id": thread_id}}
    return data_agent.invoke({"messages": [HumanMessage(content=question)]}, config)


def chat():
    thread_id = str(uuid.uuid4())        # one id = one conversation
    print("Data Agent ready. Type 'exit' to quit, 'sql' to see the last SQL query.")
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if user.lower() == "sql":
            state = data_agent.get_state({"configurable": {"thread_id": thread_id}}).values
            print(f"\nLast SQL:\n{state.get('last_sql') or '(none yet)'}")
            continue
        state = ask(user, thread_id)
        print(f"\n[{state['route_response']}] Agent: {state['messages'][-1].content}")


if __name__ == "__main__":
    chat()
