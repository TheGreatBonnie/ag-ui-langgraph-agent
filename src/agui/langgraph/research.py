# ========================================
# RESEARCH NODE FUNCTIONALITY
# ========================================

from typing import Optional, List
from langchain_core.messages import AIMessage

# Import our modular components
from .state import ResearchState
from .web_search import web_search
from .report import create_detailed_report

def research_node(messages, state: Optional[ResearchState] = None):
    """
    LangGraph node function that orchestrates the complete research workflow.
    
    Step 1: Extract the user query from the message chain
    Step 2: Set processing state to active
    Step 3: Perform web search with state tracking
    Step 4: Generate detailed report from search results
    Step 5: Return AI message with the final report
    
    Args:
        messages: List of conversation messages (LangChain message objects)
        state: Optional ResearchState instance for progress tracking
        
    Returns:
        List containing AIMessage with the generated research report
    """
    # Step 1: Extract the research query from the last message
    last = messages[-1]      # Get the most recent message
    query = last.content     # Extract the text content (user's query)
    
    # Step 2: Set processing state to indicate workflow is active
    if state:
        state.set_in_progress(True)
    
    # Step 3: Perform web search with integrated state management
    search_results = web_search(query, state)
    
    # Step 4: Generate comprehensive report from search results
    report = create_detailed_report(search_results, state)
    
    # Step 5: Return the report wrapped in an AI message for LangGraph
    return [AIMessage(content=report)]