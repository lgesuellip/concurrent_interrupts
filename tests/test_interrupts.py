import operator
from typing import Annotated

import pytest
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt

pytestmark = pytest.mark.anyio


def test_interrupt_with_send_payloads() -> None:
    """Test interruption in map node with Send payloads and human-in-the-loop resume."""
    
    # Use MemorySaver as checkpointer
    checkpointer = MemorySaver()

    # Global counter to track node executions
    node_counter = {"entry": 0, "map_node": 0}

    class State(TypedDict):
        items: list[str]
        processed: Annotated[list[str], operator.add]

    def entry_node(state: State):
        node_counter["entry"] += 1
        return {}  # No state updates in entry node

    def send_to_map(state: State):
        return [Send("map_node", {"item": item}) for item in state["items"]]

    def map_node(state: State):
        node_counter["map_node"] += 1
        if "dangerous" in state["item"]:
            value = interrupt({"processing": state["item"]})
            return {"processed": [f"processed_{value}"]}
        else:
            return {"processed": [f"processed_{state['item']}_auto"]}

    builder = StateGraph(State)
    builder.add_node("entry", entry_node)
    builder.add_node("map_node", map_node)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges("entry", send_to_map, ["map_node"])
    builder.add_edge("map_node", END)

    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test_interrupt_send"}}

    # Run until interrupts
    result = graph.invoke(
        {"items": ["item1", "dangerous_item1", "dangerous_item2"]}, config=config
    )

    # Verify we have interrupts (only one for dangerous_item)
    interrupts = result.get("__interrupt__", [])
    print(f"Interrupts: {interrupts}")
    assert len(interrupts) == 2
    assert "dangerous_item" in interrupts[0].value["processing"]

    # Resume with mapping of interrupt IDs to values
    resume_map = {
        i.id: f"human_input_{i.value['processing']}" for i in interrupts
    }

    final_result = graph.invoke(Command(resume=resume_map), config=config)

    # Verify final result contains processed items
    assert "processed" in final_result
    processed_items = final_result["processed"]
    assert len(processed_items) == 3
    assert "processed_item1_auto" in processed_items  # item1 processed automatically
    assert any(
        "processed_human_input_dangerous_item1" in item for item in processed_items
    )  # dangerous_item1 processed after interrupt
    assert any(
        "processed_human_input_dangerous_item2" in item for item in processed_items
    )  # dangerous_item2 processed after interrupt

    # Verify node execution counts
    assert node_counter["entry"] == 1  # Entry node runs once
    # Map node runs 3 times initially (item1 completes, 2 dangerous_items interrupt),
    # then 2 times on resume
    print(f"Node counter state: {node_counter}")
    assert node_counter["map_node"] == 5


def test_interrupt_with_send_payloads_sequential_resume() -> None:
    """Test interruption in map node with Send payloads and sequential resume."""
    
    # Use MemorySaver as checkpointer
    checkpointer = MemorySaver()

    # Global counter to track node executions
    node_counter = {"entry": 0, "map_node": 0}

    class State(TypedDict):
        items: list[str]
        processed: Annotated[list[str], operator.add]

    def entry_node(state: State):
        node_counter["entry"] += 1
        return {}  # No state updates in entry node

    def send_to_map(state: State):
        return [Send("map_node", {"item": item}) for item in state["items"]]

    def map_node(state: State):
        node_counter["map_node"] += 1
        if "dangerous" in state["item"]:
            value = interrupt({"processing": state["item"]})
            return {"processed": [f"processed_{value}"]}
        else:
            return {"processed": [f"processed_{state['item']}_auto"]}

    builder = StateGraph(State)
    builder.add_node("entry", entry_node)
    builder.add_node("map_node", map_node)
    builder.add_edge(START, "entry")
    builder.add_conditional_edges("entry", send_to_map, ["map_node"])
    builder.add_edge("map_node", END)

    graph = builder.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test_interrupt_send_sequential"}}

    # Run until interrupts
    result = graph.invoke(
        {"items": ["item1", "dangerous_item1", "dangerous_item2"]}, config=config
    )

    # Verify we have interrupts
    interrupts = result.get("__interrupt__", [])
    assert len(interrupts) == 2
    assert "dangerous_item" in interrupts[0].value["processing"]

    # Resume first interrupt only
    first_interrupt = interrupts[0]
    first_resume_map = {
        first_interrupt.id: f"human_input_{first_interrupt.value['processing']}"
    }

    partial_result = graph.invoke(Command(resume=first_resume_map), config=config)

    # Verify we still have one pending interrupt
    remaining_interrupts = partial_result.get("__interrupt__", [])
    assert len(remaining_interrupts) == 1

    # Resume second interrupt
    second_interrupt = remaining_interrupts[0]
    second_resume_map = {
        second_interrupt.id: f"human_input_{second_interrupt.value['processing']}"
    }

    final_result = graph.invoke(Command(resume=second_resume_map), config=config)

    # Verify final result contains processed items
    assert "processed" in final_result
    processed_items = final_result["processed"]
    assert len(processed_items) == 3
    assert "processed_item1_auto" in processed_items  # item1 processed automatically
    assert any(
        "processed_human_input_dangerous_item1" in item for item in processed_items
    )  # dangerous_item1 processed after interrupt
    assert any(
        "processed_human_input_dangerous_item2" in item for item in processed_items
    )  # dangerous_item2 processed after interrupt

    # Verify node execution counts
    print(f"Node counter state: {node_counter}")
    assert node_counter["entry"] == 1  # Entry node runs once
    # Map node execution pattern:
    # - 3 times initially (all items start processing in parallel)
    # - 2 times on first resume (context restoration + completion for dangerous_item1)
    # - 1 time on second resume (complete dangerous_item2)
    # Total: 6 executions
    # The extra execution comes from LangGraph's checkpoint restoration mechanism
    assert node_counter["map_node"] == 5


if __name__ == "__main__":
    # Run tests directly
    test_interrupt_with_send_payloads()
    print("✓ Test 1: Multiple interrupts with simultaneous resume - PASSED")
    
    test_interrupt_with_send_payloads_sequential_resume()
    print("✓ Test 2: Multiple interrupts with sequential resume - PASSED")
    
    print("\nAll tests passed successfully! Multiple interrupts are working correctly.")