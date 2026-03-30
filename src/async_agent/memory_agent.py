import os
import asyncio
from typing import List, AsyncIterator
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

# Load environment variables
load_dotenv()

# We define a custom tool to show integration
class AsyncAgentSystem:
    """
    Demonstrates a non-blocking, asynchronous agentic loop.
    This pattern is vital for low-latency chat interfaces (like WebSockets / Server-Sent Events).
    Also includes a Token/Buffer-Window Memory to prevent context window exhaustion over long conversations.
    """
    
    def __init__(self):
        # Tools initialization
        self.tools = [WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())]
        
        # LLM Initialization
        # We use a tool-calling capable model
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Define the system prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful, asynchronous AI assistant. Use your available tools if you don't know the answer. "
                       "You have access to Wikipedia. Summarize answers concisely."),
            # This placeholder holds the dynamic chat history (memory)
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            # This placeholder holds intermediate tool-calling steps
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create Memory: Retain only the last 3 conversational turns to save tokens
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            return_messages=True,
            k=3 
        )
        
        # Construct the core agent and its executor
        agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
        
        # The executors handles the while-loop of thoughts and tool iterations
        self.agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=False,
            # We don't use the executor's built-in memory arg here if we are managing it manually
            # But AgentExecutor also accepts memory directly
            memory=self.memory,
            handle_parsing_errors=True
        )

    async def chat_stream(self, user_input: str) -> AsyncIterator[str]:
        """
        Asynchronously streams the execution logs and final response of the agent.
        """
        print(f"\n[USER INPUT]: {user_input}")
        
        # Using astream_events to get granular observability over the async execution
        # (This is the advanced way to stream in Langchain > 0.2)
        async for event in self.agent_executor.astream_events(
            {"input": user_input},
            version="v1" # Required for API stability
        ):
            kind = event["event"]
            
            # 1. When the agent uses a tool
            if kind == "on_tool_start":
                tool_name = event["name"]
                tool_input = event["data"].get("input")
                print(f"[AGENT TOOL EXECUTION]: Using `{tool_name}` with input: {tool_input}")
            
            # 2. Streaming the final model response token-by-token
            elif kind == "on_chat_model_stream":
                # Ensure we only stream the final output, not the tool calling reasoning
                if "chunk" in event["data"]:
                    content = event["data"]["chunk"].content
                    if content and isinstance(content, str):
                        print(content, end="", flush=True)

        print() # Newline after stream finishes

# ==========================================
# Async Execution Entry Point
# ==========================================
async def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment. The script requires it to run.")
        return

    print("Initializing Asynchronous Memory Agent...")
    app = AsyncAgentSystem()
    
    # Simulate a conversation
    turn_1 = "Hi! Who won the Nobel Prize in Physics in 2023?"
    await app.chat_stream(turn_1)
    
    # Testing the sliding window memory
    print("\n--- Next Turn ---")
    turn_2 = "What were they awarded it for?" 
    await app.chat_stream(turn_2)
    
    print("\n--- Final Turn ---")
    turn_3 = "Can you summarize our conversation so far?"
    await app.chat_stream(turn_3)
    
    print("\n======================\n")
    print(f"Memory Buffer State (Should only have recent msg due to Window k=3):\n{app.memory.load_memory_variables({})}")

if __name__ == "__main__":
    asyncio.run(main())
