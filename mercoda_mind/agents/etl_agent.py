import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END

from models.schema import ETLAgentSchema
from utils import etl_tools, pipelines
from utils.llm_pickup import pick_llm

MAX_LLM_STEPS = 10        # last-resort limit: AI calls per user message (normal work uses 2-7)
MAX_FAILED_ATTEMPTS = 3   # the same tool may fail at most this many times per user message
KEEP_TURNS = 4            # only the last 4 user messages (and what followed) are sent to the AI


# ------------------------------------ AGENT TOOLS ------------------------------------ #
# The docstrings are what the AI reads to decide which tool to use, so keep them clear.

@tool
def list_files_tool(folder: str = "data") -> str:
    """List the csv/json/parquet data files available in a folder (default: data).
    Use this when the user mentions a file but not its exact path."""
    return etl_tools.list_files(folder)


@tool
def profile_file_tool(path: str) -> str:
    """Show a data file's real columns, data types, null counts and first rows.
    ALWAYS call this before writing a transform, so you use real column names."""
    return etl_tools.profile_file(path)


@tool
def extract_api_tool(url: str, output_name: str = "extracted_data", format: str = "csv",
                     records_key: str = "", max_pages: int = 20, refresh: bool = False) -> str:
    """Get JSON data from an API URL as a table in data/extract/<output_name>.<format>.
    Follows 'next' page links automatically (up to max_pages pages).
    If the same URL was downloaded in the last 24 hours, the existing file is REUSED and
    nothing is downloaded.

    Args:
        url: The API endpoint, e.g. https://pokeapi.co/api/v2/pokemon?limit=100
        output_name: File name without extension, e.g. 'pokemon'
        format: csv, json or parquet
        records_key: Name of the list inside the JSON that holds the rows (e.g. 'results').
                     Leave empty to detect it automatically.
        max_pages: Maximum number of pages to follow.
        refresh: true ONLY if the user asks for fresh / latest / re-downloaded data.
    """
    return etl_tools.extract_api(url, output_name, format, records_key or None, max_pages, refresh)


@tool
def run_transform_tool(input_path: str, output_path: str, transform_code: str, request: str) -> str:
    """Transform a data file with pandas and save the result.

    Args:
        input_path: Source file, e.g. data/extract/pokemon.csv
        output_path: Where to save, e.g. data/transform/grass_pokemon.csv
                     (the extension .csv/.json/.parquet sets the format)
        transform_code: ONLY the body of a function `transform(df)` that returns a DataFrame,
                        e.g.  return df[df["type"] == "grass"]
                        `pd` (pandas) and `np` (numpy) are available. Do NOT read or save files,
                        do NOT import other libraries.
        request: The user's request in a few words, e.g. 'keep only grass pokemon'
    """
    result = etl_tools.run_transform(input_path, output_path, transform_code)
    if not result["ok"]:
        return result["message"]
    run_id = pipelines.record_run(request, result)
    return (f"{result['message']}\n\nThis was recorded as draft run #{run_id}. "
            f"It is NOT saved as a pipeline unless the user asks.")


@tool
def list_runs_tool() -> str:
    """List recent transform runs (#1, #2, ...) and whether each is a draft or saved."""
    return pipelines.list_runs()


@tool
def save_pipeline_tool(run_ref: str, name: str) -> str:
    """Save a draft run as a reusable pipeline script. ONLY use when the user asks to save.

    Args:
        run_ref: Which run: 'this'/'last' (newest), 'previous', '#2', or words from the
                 request such as 'grass'.
        name: Short pipeline name, e.g. 'grass_pokemon'
    """
    try:
        return pipelines.save_run(run_ref, name)
    except ValueError as e:
        return f"ERROR: {e}"


@tool
def list_pipelines_tool() -> str:
    """List saved pipelines."""
    return pipelines.list_pipelines()


@tool
def run_pipeline_tool(name: str, input_path: str = "", output_path: str = "") -> str:
    """Re-run a saved pipeline (no code writing needed). Optionally on a new input file
    and/or to a new output file."""
    return pipelines.run_pipeline(name, input_path, output_path)


tools = [list_files_tool, profile_file_tool, extract_api_tool, run_transform_tool,
         list_runs_tool, save_pipeline_tool, list_pipelines_tool, run_pipeline_tool]
tools_by_name = {t.name: t for t in tools}

llm = pick_llm("medium")
# gpt-5.6-terra only supports function tools on /v1/chat/completions with reasoning turned off
llm_bind = llm.bind_tools(tools, reasoning_effort="none")


SYSTEM_PROMPT = """You are an ETL assistant for data analysts. You extract data from APIs,
transform files with pandas, and manage reusable pipelines, using your tools.

Rules:
0. Reuse data you already have. If the user asks about data (e.g. "pokemon") without giving
   a URL, call list_files_tool first: it shows each file's source URL and age. Work on the
   existing file instead of downloading again. Only set refresh=true when the user asks for
   fresh or latest data.
1. Before writing a transform, call profile_file_tool on the input file. Use only column
   names that really exist. If the data cannot answer the request (e.g. a needed column is
   missing), say so plainly and suggest what data would be needed.
2. For transforms, call run_transform_tool with only the body of transform(df).
   Never overwrite the input file; write results to data/transform/.
3. If a tool returns ERROR, read the message, fix the problem and try again.
   Stop after 3 failed attempts on the same task and explain the problem to the user.
4. Report only facts from tool results: row counts, columns, file paths.
   Never say something worked if a tool returned ERROR.
5. Successful transforms become draft runs. Call save_pipeline_tool ONLY when the user
   asks to save. If it is unclear which run they mean, call list_runs_tool and ask.
6. Paths are relative to the project folder, e.g. data/extract/pokemon.csv.
7. Keep the final answer short: what was done, the numbers, where the file is.
"""


# ---------------------------------------- AGENT GRAPH ---------------------------------------- #

def recent_turns(messages: list, keep_turns: int = KEEP_TURNS) -> list:
    """Keep only the last `keep_turns` user messages and everything after them.
    Cutting at a HumanMessage is always safe: a tool call and its result are never split apart.
    Older details are not lost: files, runs and pipelines are on disk, and the AI can
    look them up with list_files_tool, list_runs_tool and list_pipelines_tool."""
    human_positions = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if len(human_positions) <= keep_turns:
        return messages
    return messages[human_positions[-keep_turns]:]


def failures_this_turn(messages: list, tool_name: str) -> int:
    """How many times `tool_name` returned ERROR since the user's latest message."""
    start = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    turn = messages[start:]
    call_names = {c["id"]: c["name"] for m in turn if isinstance(m, AIMessage) for c in m.tool_calls}
    return sum(1 for m in turn
               if isinstance(m, ToolMessage)
               and call_names.get(m.tool_call_id) == tool_name
               and m.content.startswith("ERROR"))


def llm_node(state: ETLAgentSchema):
    if state.step_count >= MAX_LLM_STEPS:
        return {"messages": [AIMessage(content=(
            f"I stopped after {MAX_LLM_STEPS} steps without finishing. "
            f"Please make the request more specific or split it into smaller steps."))]}

    # Send the real message list (not an f-string), so the AI sees a proper chat
    # with each tool call linked to its result. Only recent turns, to keep cost down.
    history = recent_turns(state.messages)
    reply = llm_bind.invoke([SystemMessage(content=SYSTEM_PROMPT)] + history)
    return {"messages": [reply], "step_count": state.step_count + 1}


def tool_node(state: ETLAgentSchema):
    results = []
    for call in state.messages[-1].tool_calls:
        tool_fn = tools_by_name.get(call["name"])
        if failures_this_turn(state.messages, call["name"]) >= MAX_FAILED_ATTEMPTS:
            # Enforced in code: the tool is not run again, whatever the AI decides.
            output = (f"STOP: {call['name']} already failed {MAX_FAILED_ATTEMPTS} times for this request, "
                      f"so it was not run again. Do not call it again. Tell the user what went wrong "
                      f"and what they could change.")
        elif tool_fn is None:
            output = f"ERROR: there is no tool called '{call['name']}'. Available: {list(tools_by_name)}"
        else:
            try:
                output = tool_fn.invoke(call["args"])
            except Exception as e:          # one broken tool call must not crash the agent
                output = f"ERROR: {call['name']} failed: {type(e).__name__}: {e}"
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}


def is_tool_call(state: ETLAgentSchema):
    last = state.messages[-1]
    return "tool_node" if getattr(last, "tool_calls", None) else "end"


etl_analyst_graph = StateGraph(ETLAgentSchema)
etl_analyst_graph.add_node("llm_node", llm_node)
etl_analyst_graph.add_node("tool_node", tool_node)
etl_analyst_graph.add_edge(START, "llm_node")
etl_analyst_graph.add_conditional_edges("llm_node", is_tool_call,
                                        {"tool_node": "tool_node", "end": END})
etl_analyst_graph.add_edge("tool_node", "llm_node")

etl_analyst = etl_analyst_graph.compile()


def chat():
    """Simple multi-turn chat, so follow-ups like 'save this' work."""
    messages = []
    print("ETL Analyst ready. Type 'exit' to quit.")
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        result = etl_analyst.invoke({"messages": messages + [HumanMessage(content=user)],
                                     "step_count": 0})
        messages = result["messages"]
        print(f"\nAgent: {messages[-1].content}")


if __name__ == "__main__":
    # python agents/etl_agent.py          -> save graph image + run 2 example requests
    # python agents/etl_agent.py --chat   -> chat with the ETL agent yourself
    from utils.graph_image import save_graph
    save_graph(etl_analyst, "etl_agent_graph")

    if "--chat" in sys.argv:
        chat()
    else:
        examples = [
            "Extract the data from https://pokeapi.co/api/v2/pokemon?limit=200 "
            "and save it as pokemon in csv format",
            "From data/extract/pokemon.csv keep only bulbasaur and save it to "
            "data/transform/bulbasaur.csv",
        ]
        messages = []
        for question in examples:
            result = etl_analyst.invoke({"messages": messages + [HumanMessage(content=question)],
                                         "step_count": 0})
            print("\n" + "#" * 80)
            for message in result["messages"][len(messages):]:   # only this request's messages
                message.pretty_print()
            messages = result["messages"]                          # keep history for the next request