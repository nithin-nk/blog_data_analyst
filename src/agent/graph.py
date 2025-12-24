"""
Graph module - LangGraph StateGraph definition for the blog agent.

This module defines the blog agent graph with:
- Routing functions for conditional edges
- build_blog_agent_graph() to create the compiled graph

Graph structure:
    topic_discovery → content_landscape_analysis → planning → research →
    validate_sources → write_section ↓ (loop) → final_assembly → END
"""

from langgraph.graph import END, StateGraph

from .nodes import (
    content_landscape_analysis_node,
    final_assembly_node,
    planning_node,
    research_node,
    topic_discovery_node,
    validate_sources_node,
    write_section_node,
)
from .state import BlogAgentState


def section_router(state: BlogAgentState) -> str:
    """
    Route after write_section_node: continue writing or move to assembly.

    Returns:
        "write_next" if more sections to write, "all_complete" otherwise
    """
    plan = state.get("plan", {})
    sections = plan.get("sections", [])
    current_idx = state.get("current_section_index", 0)

    # Only count non-optional sections
    required_sections = [s for s in sections if not s.get("optional")]

    if current_idx < len(required_sections):
        return "write_next"
    return "all_complete"


def build_blog_agent_graph() -> StateGraph:
    """
    Build the blog agent graph with section loop.

    Graph structure:
        topic_discovery → content_landscape_analysis → planning → research →
        validate_sources → write_section ↓ (loop) → final_assembly → END

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create graph with BlogAgentState
    graph = StateGraph(BlogAgentState)

    # Add all nodes
    graph.add_node("topic_discovery", topic_discovery_node)
    graph.add_node("content_landscape_analysis", content_landscape_analysis_node)
    graph.add_node("planning", planning_node)
    graph.add_node("research", research_node)
    graph.add_node("validate_sources", validate_sources_node)
    graph.add_node("write_section", write_section_node)
    graph.add_node("final_assembly", final_assembly_node)

    # Set entry point
    graph.set_entry_point("topic_discovery")

    # Linear edges for pipeline
    graph.add_edge("topic_discovery", "content_landscape_analysis")
    graph.add_edge("content_landscape_analysis", "planning")
    graph.add_edge("planning", "research")
    graph.add_edge("research", "validate_sources")
    graph.add_edge("validate_sources", "write_section")

    # Section loop: write_section routes back to itself or to assembly
    graph.add_conditional_edges(
        "write_section",
        section_router,
        {
            "write_next": "write_section",
            "all_complete": "final_assembly",
        },
    )

    # End after assembly (no human review yet in Round 1)
    graph.add_edge("final_assembly", END)

    return graph.compile()
