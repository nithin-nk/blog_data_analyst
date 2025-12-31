"""
Graph module - LangGraph StateGraph definition for the blog agent.

This module defines the blog agent graph with:
- Routing functions for conditional edges
- build_blog_agent_graph() to create the compiled graph

Graph structure:
    topic_discovery → content_landscape_analysis → planning → preview_validation →
    (conditional: pass → section_selection OR fail → planning) → section_selection → research →
    validate_sources → write_section ↓ (loop) → final_assembly → human_review → END
"""

from langgraph.graph import END, StateGraph

from .nodes import (
    content_landscape_analysis_node,
    final_assembly_node,
    human_review_node,
    planning_node,
    preview_validation_node,
    research_node,
    section_selection_node,
    topic_discovery_node,
    validate_sources_node,
    write_section_node,
)
from .state import BlogAgentState, Phase


def preview_validation_router(state: BlogAgentState) -> str:
    """
    Route after preview_validation_node.

    Returns:
        "planning" if validation failed (trigger replanning)
        "section_selection" if validation passed (proceed to section selection)
    """
    current_phase = state.get("current_phase", "")

    # If phase went back to PLANNING, validation failed and we need to replan
    if current_phase == Phase.PLANNING.value:
        return "planning"

    # If phase advanced to SECTION_SELECTION, validation passed
    if current_phase == Phase.SECTION_SELECTION.value:
        return "section_selection"

    # Default: proceed to section selection (shouldn't happen but safe fallback)
    return "section_selection"


def section_router(state: BlogAgentState) -> str:
    """
    Route after write_section_node: continue writing or move to assembly.

    Returns:
        "write_next" if more sections to write, "all_complete" otherwise
    """
    plan = state.get("plan", {})
    sections = plan.get("sections", [])
    current_idx = state.get("current_section_index", 0)

    # Filter to selected sections (or fallback to required only)
    selected_ids = state.get("selected_section_ids", [])
    if selected_ids:
        # Use user selection
        required_sections = [s for s in sections if s["id"] in selected_ids]
    else:
        # Fallback: filter out optional (backward compatibility)
        required_sections = [s for s in sections if not s.get("optional")]

    if current_idx < len(required_sections):
        return "write_next"
    return "all_complete"


def review_router(state: BlogAgentState) -> str:
    """
    Route after human_review_node.

    Returns:
        END for both approve and quit (future slices may add edit paths)
    """
    # Both approve and quit end the pipeline for now
    # Future slices may add edit/retry paths
    return END


def build_blog_agent_graph() -> StateGraph:
    """
    Build the blog agent graph with preview validation, section selection, section loop, and human review.

    Graph structure:
        topic_discovery → content_landscape_analysis → planning → preview_validation →
        (conditional: pass → section_selection OR fail → planning) → section_selection → research →
        validate_sources → write_section ↓ (loop) → final_assembly → human_review → END

    Returns:
        Compiled LangGraph StateGraph
    """
    # Create graph with BlogAgentState
    graph = StateGraph(BlogAgentState)

    # Add all nodes
    graph.add_node("topic_discovery", topic_discovery_node)
    graph.add_node("content_landscape_analysis", content_landscape_analysis_node)
    graph.add_node("planning", planning_node)
    graph.add_node("preview_validation", preview_validation_node)
    graph.add_node("section_selection", section_selection_node)
    graph.add_node("research", research_node)
    graph.add_node("validate_sources", validate_sources_node)
    graph.add_node("write_section", write_section_node)
    graph.add_node("final_assembly", final_assembly_node)
    graph.add_node("human_review", human_review_node)

    # Set entry point
    graph.set_entry_point("topic_discovery")

    # Linear edges for pipeline
    graph.add_edge("topic_discovery", "content_landscape_analysis")
    graph.add_edge("content_landscape_analysis", "planning")
    graph.add_edge("planning", "preview_validation")

    # Preview validation conditional routing
    graph.add_conditional_edges(
        "preview_validation",
        preview_validation_router,
        {
            "planning": "planning",  # Failed validation → replan
            "section_selection": "section_selection",  # Passed validation → section selection
        },
    )

    # Section selection → research
    graph.add_edge("section_selection", "research")

    # Continue pipeline after section selection
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

    # Assembly → Human Review → END
    graph.add_edge("final_assembly", "human_review")
    graph.add_conditional_edges("human_review", review_router)

    return graph.compile()
