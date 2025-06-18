
# Standard library imports
import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, List

# Third-party imports
from dotenv import load_dotenv  # Environment variable management
load_dotenv()  # Load environment variables from .env file
from fastapi import FastAPI, Request  # Web framework
from fastapi.responses import StreamingResponse  # For streaming responses
from pydantic import BaseModel  # For data validation

# AG-UI protocol components for communication with frontend
from ag_ui.core import (
  RunAgentInput,   # Represents the input to an agent run
  Message,         # Represents a message in the conversation
  EventType,       # Enum of event types used in the protocol
  RunStartedEvent, # Event signaling the start of an agent run
  RunFinishedEvent,# Event signaling the end of an agent run
  TextMessageStartEvent,   # Event signaling the start of a text message
  TextMessageContentEvent, # Event carrying the content of a text message
  TextMessageEndEvent      # Event signaling the end of a text message
)
from ag_ui.encoder import EventEncoder  # Encodes events to Server-Sent Events format

# LangGraph and LangChain components for the research agent
from langgraph.graph import Graph
from langchain_core.messages import AIMessage, HumanMessage

# Local research agent components
from src.agui.langgraph.agent import build_research_graph, ResearchState

# Create FastAPI application
app = FastAPI(title="AG-UI Endpoint")

@app.post("/")
async def langgraph_research_endpoint(input_data: RunAgentInput):
    """
    LangGraph-based research processing endpoint with integrated state management.
    """
    async def event_generator():
        """
        Asynchronous generator that produces a stream of AG-UI protocol events.
        """
        # Create an event encoder to properly format SSE events
        encoder = EventEncoder()
        
        # Extract the research query from the most recent message
        query = input_data.messages[-1].content
        message_id = str(uuid.uuid4())  # Generate a unique ID for this message
        
        print(f"[DEBUG] LangGraph Research started with query: {query}")

        # Signal the start of the agent run
        yield encoder.encode(
            RunStartedEvent(
                type=EventType.RUN_STARTED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id
            )
        )

        # Create a list to collect emitted events
        emitted_events = []
        
        def event_emitter(encoded_event):
            """Callback function to collect events from the research state."""
            emitted_events.append(encoded_event)
        
        # Create research state with event emitter
        research_state = ResearchState(
            message_id=message_id,
            query=query,
            event_emitter=event_emitter
        )
        
        # Emit initial state snapshot
        research_state.emit_snapshot()
        
        # Yield any events that were emitted during initialization
        for event in emitted_events:
            yield event
        emitted_events.clear()
        
        # Build the research graph with state management
        graph = build_research_graph(research_state)
        
        print(f"[DEBUG] Executing LangGraph workflow with state management")
        
        # Execute the LangGraph workflow with the query
        result = graph.invoke([HumanMessage(content=query)])
        
        # Yield any events that were emitted during processing
        for event in emitted_events:
            yield event
        emitted_events.clear()
        
        print(f"[DEBUG] LangGraph invoke API succeeded")
        print(f"[DEBUG] LangGraph result type: {type(result)}, content: {str(result)[:100]}...")
        
        # Get the report from the AI message content
        print(f"[DEBUG] Result is a list with {len(result)} items")
        report_item = result[0]
        print(f"[DEBUG] First item type: {type(report_item)}")
        
        report_content = report_item.content
        print(f"[DEBUG] Report content extracted, length: {len(report_content)}")
        
        # Send the text message with the report content
        yield encoder.encode(
            TextMessageStartEvent(
                type=EventType.TEXT_MESSAGE_START,
                message_id=message_id,
                role="assistant"
            )
        )
        
        yield encoder.encode(
            TextMessageContentEvent(
                type=EventType.TEXT_MESSAGE_CONTENT,
                message_id=message_id,
                delta=report_content
            )
        )
        
        yield encoder.encode(
            TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END,
                message_id=message_id
            )
        )

        # Complete the run
        yield encoder.encode(
            RunFinishedEvent(
                type=EventType.RUN_FINISHED,
                thread_id=input_data.thread_id,
                run_id=input_data.run_id
            )
        )

    # Return a streaming response containing SSE events from the generator
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

def main():
    """
    Entry point for running the FastAPI server.
    """
    import uvicorn
    uvicorn.run("src.agui.main:app", host="0.0.0.0", port=8000, reload=True)
 
if __name__ == "__main__":
    main()