import os
from typing import TypedDict, Annotated, List, Literal
import operator

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# ==========================================
# 1. Define the Graph State
# ==========================================
class AgentState(TypedDict):
    """
    Represents the state of our graph.
    `messages` is a list that appends new messages using the `operator.add` reducer.
    `research_iterations` prevents infinite tool loops.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    research_iterations: int
    topic: str

# ==========================================
# 2. Define the Nodes (Agents / Actions)
# ==========================================
def researcher_node(state: AgentState) -> dict:
    """
    The Researcher uses search tools to gather context about the given topic.
    """
    print("--- [NODE: Researcher] Gathering Information ---")
    
    # Initialize the LLM and bind the search tool
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    search_tool = DuckDuckGoSearchRun()
    llm_with_tools = llm.bind_tools([search_tool])
    
    # We provide a system prompt to guide the researcher
    system_message = SystemMessage(
        content="You are a meticulous researcher. Use the search tool to find detailed and accurate information about the user's topic. "
                "If you need more information, use the tool again. If you have enough, synthesize a rough summary."
    )
    
    messages = [system_message] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
        "research_iterations": state["research_iterations"] + 1
    }

def writer_node(state: AgentState) -> dict:
    """
    The Writer synthesizes the research context into a professional output.
    """
    print("--- [NODE: Writer] Synthesizing Final Output ---")
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
    
    system_message = SystemMessage(
        content="You are an expert technical writer. You will receive research data and a topic. "
                "Draft a highly professional, comprehensive 3-paragraph summary on the topic based strictly on the research provided."
    )
    
    messages = [system_message] + state["messages"]
    response = llm.invoke(messages)
    
    return {"messages": [response]}

# ==========================================
# 3. Define the Edges (Routing Logic)
# ==========================================
def supervisor_router(state: AgentState) -> Literal["tools", "writer", "__end__"]:
    """
    Determines whether to execute a tool, continue researching, or pass to the writer.
    """
    last_message = state["messages"][-1]
    
    # Condition 1: If the model decided to call a tool, route to the 'tools' node
    if last_message.tool_calls:
        print("    -> Routing to Tools...")
        return "tools"
    
    # Condition 2: Loop breaker. If we researched too much, force it to the writer
    if state["research_iterations"] >= 3:
        print("    -> Maximum research iterations hit. Forcing to Writer...")
        return "writer"
        
    # Condition 3: If no tool calls and we have information, pass to writer
    print("    -> Routing to Writer...")
    return "writer"

# ==========================================
# 4. Build the Graph
# ==========================================
def build_research_graph() -> StateGraph:
    """Constructs the deterministic state machine workflow."""
    print("[INFO] Building Research Graph...")
    workflow = StateGraph(AgentState)
    
    # Define tool node
    tools = [DuckDuckGoSearchRun()]
    tool_node = ToolNode(tools)
    
    # Add nodes to the graph
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("writer", writer_node)
    
    # Define control flow edges
    workflow.set_entry_point("researcher")
    
    workflow.add_conditional_edges(
        "researcher",
        supervisor_router,
        {
            "tools": "tools",
            "writer": "writer"
        }
    )
    
    # After tools are run, always go back to the researcher to view the results
    workflow.add_edge("tools", "researcher")
    
    # Writer is the final step
    workflow.add_edge("writer", END)
    
    # Compile Graph
    return workflow.compile()

# ==========================================
# Execution Entry Point
# ==========================================
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment. The script requires it to run.")
    else:
        app = build_research_graph()
        
        # Define the starting state
        topic = "The integration of LangGraph in deterministic LLM workflows"
        initial_state = {
            "messages": [HumanMessage(content=f"Research topic: {topic}")],
            "research_iterations": 0,
            "topic": topic
        }
        
        print(f"\n[STARTING GRAPH EXECUTION] Topic: {topic}\n")
        
        # Execute the graph
        for output in app.stream(initial_state):
            for key, value in output.items():
                print(f"Finished node: '{key}'")
                
        # Print final result
        final_message = value["messages"][-1].content
        print(f"\n==== FINAL OUTPUT ====\n{final_message}\n======================\n")
