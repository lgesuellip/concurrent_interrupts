"""
LangGraph React Agent with Concurrent Interrupts and Inbox Pattern

This module implements a React agent with human-in-the-loop interrupt capabilities
using the agent-inbox pattern. The agent supports concurrent tool execution with
individual interrupt points for each tool.

Usage Example:
    ```python
    from agent_with_inbox import build_agent
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command
    
    # Build the agent
    agent = build_agent()
    checkpointer = MemorySaver()
    
    # Configure with checkpointer for persistence
    config = {"configurable": {"thread_id": "my_session"}}
    
    # Run with interrupt handling using Command pattern
    result = agent.invoke(
        {"messages": [("user", "Analyze the sales data")]},
        config=config,
        checkpointer=checkpointer
    )
    
    # Handle interrupts using Command pattern with interrupt ID mapping
    state = agent.get_state(config, checkpointer=checkpointer)
    if state.next:
        # Resume with mapping of interrupt IDs to values
        resume_map = {
            i.interrupt_id: {"type": "accept"}
            for i in state.interrupts
        }
        result = agent.invoke(Command(resume=resume_map), config=config, checkpointer=checkpointer)
    ```

Architecture:
- Uses LangGraph's React agent with v2 parallel execution
- Tools implement structured interrupt patterns
- Each tool can interrupt independently
- Supports multiple interrupt types (approval, editing, cancellation)
"""

import time
import logging
import uuid
import os
from pathlib import Path
from datetime import datetime
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

# Load environment from parent directory
from dotenv import load_dotenv
# Load .env from parent directory
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)
logger = logging.getLogger(__name__)
logger.info(f"Loaded environment from: {env_path}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# Define Human Interrupt schema following agent-inbox pattern
class HumanInterruptConfig(BaseModel):
    """
    Configuration for human interrupt behavior and allowed actions.
    
    This class defines what actions a human can take when an interrupt occurs.
    It follows the agent-inbox pattern for structured interrupt handling.
    
    Attributes:
        allow_ignore (bool): Whether the human can ignore/cancel the action.
            Default: True. When True, humans can cancel tool execution.
        allow_respond (bool): Whether the human can provide feedback/response.
            Default: False. When True, humans can provide text feedback.
        allow_edit (bool): Whether the human can modify tool parameters.
            Default: True. When True, humans can edit action arguments.
        allow_accept (bool): Whether the human can approve the action.
            Default: True. When True, humans can approve tool execution.
    
    Example:
        ```python
        # Allow only approval or cancellation
        config = HumanInterruptConfig(
            allow_ignore=True,
            allow_respond=False,
            allow_edit=False,
            allow_accept=True
        )
        
        # Allow all actions including editing
        config = HumanInterruptConfig(
            allow_ignore=True,
            allow_respond=True,
            allow_edit=True,
            allow_accept=True
        )
        ```
    """
    allow_ignore: bool = True
    allow_respond: bool = False
    allow_edit: bool = True
    allow_accept: bool = True


class ActionRequest(BaseModel):
    """
    Represents a tool action request with parameters.
    
    This class encapsulates the details of a tool action that requires
    human approval or intervention. It contains the action name and
    the arguments that will be passed to the tool.
    
    Attributes:
        action (str): The name of the action/tool to be executed.
            Should match the function name of the tool.
        args (dict): Dictionary of arguments to pass to the tool.
            Keys should match the tool's parameter names.
    
    Example:
        ```python
        # Request for data analysis
        request = ActionRequest(
            action="fast_analysis_tool",
            args={"data": "Q3 sales figures"}
        )
        
        # Request for processing with mode
        request = ActionRequest(
            action="slow_processing_tool", 
            args={"data": "customer feedback", "mode": "full"}
        )
        ```
    """
    action: str
    args: dict


class HumanInterrupt(BaseModel):
    """
    Complete interrupt request sent to human for approval/intervention.
    
    This class represents a complete interrupt request that combines the action
    to be performed, configuration options, and descriptive information for
    the human operator. It follows the agent-inbox pattern for structured
    human-in-the-loop interactions.
    
    Attributes:
        id (str): Unique identifier for tracking this interrupt.
            Generated using UUID4 for global uniqueness.
        action_request (ActionRequest): The tool action requiring approval.
            Contains the tool name and arguments.
        config (HumanInterruptConfig): Configuration for allowed human actions.
            Defines what the human can do with this interrupt.
        description (str): Human-readable description of what approval is needed.
            Should clearly explain what the tool will do.
    
    Example:
        ```python
        import uuid
        
        # Create interrupt for data analysis approval
        interrupt = HumanInterrupt(
            id=str(uuid.uuid4()),
            action_request=ActionRequest(
                action="analyze_data",
                args={"dataset": "sales_q3.csv"}
            ),
            config=HumanInterruptConfig(
                allow_ignore=True,
                allow_edit=True,
                allow_accept=True
            ),
            description="Requesting approval to analyze Q3 sales data"
        )
        ```
    
    Response Handling:
        The interrupt response can be:
        - {"type": "accept"} - Approve the action as-is
        - {"type": "ignore"} - Cancel/skip the action
        - {"type": "edit", "args": {...}} - Modify parameters and approve
        - {"type": "response", "content": "..."} - Provide feedback
    """
    id: str  # Unique ID for tracking interrupts
    action_request: ActionRequest
    config: HumanInterruptConfig
    description: str


# Tools with concurrent interrupt capability, using Send API-V2
@tool
def fast_analysis_tool(data: str) -> str:
    """
    Performs fast data analysis with human approval interrupt.
    
    This tool simulates a quick analysis operation that requires human approval
    before proceeding. It demonstrates the interrupt pattern where tool execution
    pauses to request human intervention before completing the operation.

    Args:
        data (str): The data to be analyzed. Can be any string describing
            the dataset, query, or content to analyze.
    
    Returns:
        str: Analysis result message indicating the outcome:
            - "Fast analysis completed: {data} - Decision: approved" if accepted
            - "Fast analysis cancelled for: {data}" if ignored/cancelled
            - Result includes human decision for audit trail
    
    Raises:
        Exception: If interrupt handling fails or tool execution encounters errors.
            Errors are logged and may be propagated depending on interrupt response.
    
    Example Usage:
        ```python
        # Called automatically by agent
        result = fast_analysis_tool("Q3 sales data")
        
        # Human sees interrupt request:
        # "Fast analysis tool needs approval to complete analysis of: Q3 sales data"
        
        # Human responses:
        # {"type": "accept"} -> "Fast analysis completed: Q3 sales data - Decision: approved"
        # {"type": "ignore"} -> "Fast analysis cancelled for: Q3 sales data"
        ```
    """
    start_time = datetime.now()
    logger.info(f"[FAST_TOOL] Starting analysis of: {data}")

    # Simulate processing
    logger.info(f"[FAST_TOOL] Sleeping for 2 seconds...")
    time.sleep(2)

    # Create interrupt request following agent-inbox pattern
    interrupt_id = str(uuid.uuid4())
    request = HumanInterrupt(
        id=interrupt_id,
        action_request=ActionRequest(
            action="fast_analysis",
            args={"data": data}
        ),
        config=HumanInterruptConfig(
            allow_ignore=True,
            allow_respond=False,
            allow_edit=True,
            allow_accept=True
        ),
        description=f"Fast analysis tool needs approval to complete analysis of: {data}."
    )

    # Interrupt for human approval
    logger.info(f"[FAST_TOOL] Interrupting for human approval with ID: {interrupt_id}")
    response = interrupt(request.model_dump())
    logger.info(f"[FAST_TOOL] Received response for interrupt {interrupt_id}: {response}")

    # Handle response
    if isinstance(response, dict):
        response_type = response.get("type", "accept")
        logger.info(f"[FAST_TOOL] Response type: {response_type}")
        
        if response_type == "ignore":
            result = f"Fast analysis cancelled for: {data}"
            logger.warning(f"[FAST_TOOL] Analysis cancelled by user")
        else:
            result = f"Fast analysis completed: {data} - Decision: approved"
            logger.info(f"[FAST_TOOL] Analysis approved")
    else:
        result = f"Fast analysis completed: {data} - Decision: approved"
        logger.info(f"[FAST_TOOL] Analysis approved (default)")
    
    total_time = (datetime.now() - start_time).seconds
    logger.info(f"[FAST_TOOL] Total execution time: {total_time} seconds")
    return result


@tool  
def slow_processing_tool(data: str, mode: str = "full") -> str:
    """
    Performs comprehensive data processing with human approval and parameter editing.
    
    This tool simulates a more complex processing operation that requires human
    approval and allows parameter modification. It demonstrates advanced interrupt
    patterns including parameter editing and multiple response types.
    
    Args:
        data (str): The data to be processed. Can be any string describing
            the dataset, content, or information to process.
        mode (str, optional): Processing mode that determines operation depth.
            Defaults to "full". Common values:
            - "full": Complete comprehensive processing
            - "partial": Limited scope processing
            - "quick": Fast processing with reduced features
            - "detailed": Enhanced processing with extra analysis
    
    Returns:
        str: Processing result message indicating the outcome:
            - "Slow processing completed: {data} ({mode}) - Decision: approved" if accepted
            - "Slow processing cancelled for: {data}" if ignored/cancelled
            - "Slow processing completed: {data} ({old_mode}→{new_mode}) - Decision: modified" if edited
            - Result includes mode and human decision for audit trail
    
    Raises:
        Exception: If interrupt handling fails or processing encounters errors.
            Errors are logged and may be propagated depending on interrupt response.
    
    Example Usage:
        ```python
        # Called automatically by agent with default mode
        result = slow_processing_tool("customer feedback data")
        
        # Called with specific mode
        result = slow_processing_tool("sales data", mode="quick")
        
        # Human sees interrupt request:
        # "Slow processing tool needs approval for full processing of: customer feedback data"
        
        # Human responses:
        # {"type": "accept"} -> Proceed with current mode
        # {"type": "ignore"} -> Cancel processing
        # {"type": "edit", "args": {"mode": "quick"}} -> Change mode and proceed
        ```
    """
    start_time = datetime.now()
    logger.info(f"[SLOW_TOOL] Starting {mode} processing of: {data}")
    logger.info(f"[SLOW_TOOL] Processing mode: {mode}")
    
    # Simulate processing
    logger.info(f"[SLOW_TOOL] Sleeping for 2 seconds...")
    time.sleep(2)
    
    interrupt_id = str(uuid.uuid4())
    request = HumanInterrupt(
        id=interrupt_id,
        action_request=ActionRequest(
            action="slow_processing",
            args={"data": data, "mode": mode}
        ),
        config=HumanInterruptConfig(
            allow_ignore=True,
            allow_edit=True,
            allow_respond=False,
            allow_accept=True
        ),
        description=f"Slow processing tool needs approval for {mode} processing of: {data}."
    )
    
    # Interrupt for human approval
    logger.info(f"[SLOW_TOOL] Interrupting for human approval with ID: {interrupt_id}")
    response = interrupt(request.model_dump())
    logger.info(f"[SLOW_TOOL] Received response for interrupt {interrupt_id}: {response}")
    
    # Handle response
    if isinstance(response, dict):
        response_type = response.get("type", "accept")
        logger.info(f"[SLOW_TOOL] Response type: {response_type}")
        
        if response_type == "ignore":
            result = f"Slow processing cancelled for: {data}"
            logger.warning(f"[SLOW_TOOL] Processing cancelled by user")
        elif response_type == "edit":
            # Get updated mode if edited
            new_mode = response.get("args", {}).get("mode", mode)
            result = f"Slow processing completed: {data} ({new_mode}) - Decision: modified"
            logger.info(f"[SLOW_TOOL] Processing mode changed from '{mode}' to '{new_mode}'")
        else:
            result = f"Slow processing completed: {data} ({mode}) - Decision: approved"
            logger.info(f"[SLOW_TOOL] Processing approved")
    else:
        result = f"Slow processing completed: {data} ({mode}) - Decision: approved"
        logger.info(f"[SLOW_TOOL] Processing approved (default)")
    
    total_time = (datetime.now() - start_time).seconds
    logger.info(f"[SLOW_TOOL] Total execution time: {total_time} seconds")
    return result


def build_agent(local_checkpointer=False):
    """
    Build a React agent with concurrent interrupt capability and human-in-the-loop support.
    """
    model = ChatOpenAI(model="gpt-4o-mini")
    
    # Create the React agent with tools
    # Note: v2 uses Send API for parallel execution but requires specific setup
    try:
        if local_checkpointer:
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
        else:
            checkpointer = None
        # Try v2 first for parallel execution
        agent = create_react_agent(
            model=model,
            tools=[fast_analysis_tool, slow_processing_tool],
            version="v2",
            checkpointer=checkpointer
        )
        logger.info("Created React Agent v2 with parallel execution support")
    except Exception as e:
        logger.warning(f"Failed to create v2 agent: {e}")
        logger.info("Falling back to v1 agent")
    
    return agent


# For LangGraph API registration
# This creates the agent instance that can be deployed via LangGraph Functional API (langgraph up)
# The graph variable is automatically discovered by the LangGraph deployment system
graph = build_agent()


# Example usage for development and testing
if __name__ == "__main__":
    """
    Example usage of the agent with interrupt handling.
    
    This demonstrates how to use the agent in a development environment
    with proper interrupt handling and state management.
    """
    
    agent = build_agent(local_checkpointer=True)
    
    config = {
        "configurable": {"thread_id": "development_session"},
        "recursion_limit": 50
    }
    
    print("🤖 Agent with Inbox - Development Example")
    print("=" * 50)
    
    # Example messages that will trigger tools with interrupts
    example_queries = [
        "Run fast_analysis_tool on 'dataset-A' and slow_processing_tool on 'dataset-B' concurrently",
    ]
    
    for i, query in enumerate(example_queries, 1):
        print(f"\nExample {i}: {query}")
        print("-" * 40)
        
        try:
            # Initial invocation
            response = agent.invoke(
                {"messages": [("user", query)]},
                config=config,
            )

            time.sleep(4)
            
            state = agent.get_state(config)
            
            if state.next:
                print("⏸️  Interrupt occurred! In a real application:")
                print("   1. Present interrupt details to user")
                print("   2. Collect user response")  
                print("   3. Update state and resume")
                print("\n   Simulating auto-approval...")
                
                # Simulate approval using Command pattern with interrupt ID mapping
                from langgraph.types import Command
                
                # Resume with mapping of interrupt IDs to values
                resume_map = {
                    i.id: {"type": "accept"}
                    for i in state.interrupts
                }
                final_response = agent.invoke(
                    Command(resume=resume_map), 
                    config=config,
                )
                print(final_response["messages"][-1].content)
                
                print("✅ Execution completed after approval")
            else:
                print("✅ Execution completed without interrupts")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            
        print()
    
    print("Development example completed!")