"""
CLI module for the Blog Agent.

Usage:
    python -m src.agent start --title "..." --context "..."
    python -m src.agent resume <job_id>
    python -m src.agent jobs [--status complete|incomplete]
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.status import Status
from rich.table import Table

# Load .env file from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from .config import GEMINI_PRICING
from .graph import build_blog_agent_graph
from .key_manager import KeyManager
from .state import JobManager, Phase

console = Console()

# Phase display messages
PHASE_MESSAGES = {
    "topic_discovery": ("🔍", "Discovering topic and generating search queries..."),
    "planning": ("📋", "Creating blog outline and section plan..."),
    "researching": ("🔬", "Researching sources from the web..."),
    "validating_sources": ("✓", "Validating source quality and relevance..."),
    "writing": ("✍️", "Writing sections..."),
    "assembling": ("📦", "Assembling final blog post..."),
    "reviewing": ("👀", "Final review..."),
}


def _get_phase_message(phase: str, section_info: str = "") -> str:
    """Get display message for a phase."""
    icon, msg = PHASE_MESSAGES.get(phase, ("⏳", f"Processing {phase}..."))
    if section_info:
        return f"{icon} {msg} {section_info}"
    return f"{icon} {msg}"


def display_metrics_summary(state: dict[str, Any]) -> None:
    """
    Display execution metrics summary with Rich table.

    Shows timing, API calls, token usage, and costs per node.
    Also displays API key usage statistics.
    """
    metrics = state.get("metrics", {})

    if not metrics:
        return

    # Build metrics table
    table = Table(title="📊 Execution Metrics")
    table.add_column("Node", style="cyan")
    table.add_column("Duration", style="green", justify="right")
    table.add_column("API Calls", style="yellow", justify="right")
    table.add_column("Tokens (in/out)", style="blue", justify="right")
    table.add_column("Est. Cost", style="red", justify="right")

    total_cost = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    total_calls = 0
    total_duration = 0.0

    # Sort nodes by typical execution order
    node_order = [
        "topic_discovery",
        "content_landscape",
        "planning",
        "research",
        "validate_sources",
        "write_section",
        "final_assembly",
        "human_review",
    ]

    for node_name in node_order:
        if node_name not in metrics:
            continue

        data = metrics[node_name]
        duration = data.get("duration_s", 0)
        calls = data.get("calls", 0)
        tokens_in = data.get("tokens_in", 0)
        tokens_out = data.get("tokens_out", 0)
        cost = data.get("cost", 0)

        total_duration += duration
        total_calls += calls
        total_tokens_in += tokens_in
        total_tokens_out += tokens_out
        total_cost += cost

        # Format display name
        display_name = node_name.replace("_", " ").title()

        # Format duration
        if duration >= 60:
            duration_str = f"{duration / 60:.1f}m"
        else:
            duration_str = f"{duration:.1f}s"

        # Format tokens
        tokens_str = f"{tokens_in:,} / {tokens_out:,}" if tokens_in or tokens_out else "-"

        # Format cost
        cost_str = f"${cost:.4f}" if cost > 0 else "-"

        # Format calls
        calls_str = str(calls) if calls > 0 else "-"

        table.add_row(display_name, duration_str, calls_str, tokens_str, cost_str)

    # Add totals row
    total_duration_str = f"{total_duration / 60:.1f}m" if total_duration >= 60 else f"{total_duration:.1f}s"
    table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{total_duration_str}[/bold]",
        f"[bold]{total_calls}[/bold]",
        f"[bold]{total_tokens_in:,} / {total_tokens_out:,}[/bold]",
        f"[bold]${total_cost:.4f}[/bold]",
        style="bold",
    )

    console.print()
    console.print(table)

    # Display API key usage
    try:
        key_manager = KeyManager.from_env()
        usage_stats = key_manager.get_usage_stats()

        console.print("\n[bold]🔑 API Key Usage:[/bold]")
        for key_name, stats in usage_stats.get("keys", {}).items():
            requests = stats.get("requests", 0)
            remaining = stats.get("remaining", 0)
            rate_limited = stats.get("rate_limited", False)

            if rate_limited:
                status = "[red]rate-limited[/red]"
            else:
                status = f"[green]{remaining} remaining[/green]"

            console.print(f"  {key_name}: {requests} calls, {status}")
    except Exception:
        # Silently skip if key manager fails
        pass


@click.group()
def cli():
    """Blog Agent - AI-powered technical blog writer."""
    pass


@cli.command()
@click.option("--title", required=True, help="Blog title")
@click.option("--context", required=True, help="Context and notes for the blog")
@click.option(
    "--length",
    type=click.Choice(["short", "medium", "long"]),
    default="medium",
    help="Target blog length",
)
def start(title: str, context: str, length: str):
    """Start a new blog generation job."""
    asyncio.run(_run_start(title, context, length))


async def _run_start(title: str, context: str, length: str):
    """Execute the blog generation pipeline with real-time progress updates."""
    console.print(f"\n[bold blue]🚀 Blog Agent[/bold blue]")
    console.print(f"[dim]Title:[/dim] {title}")
    console.print(f"[dim]Length:[/dim] {length}\n")

    # Create or resume job
    job_manager = JobManager()
    job_id = job_manager.slugify(title)

    # Check if job already exists
    existing_state = job_manager.load_state(job_id)

    if existing_state and existing_state.get("current_phase") not in [Phase.DONE.value, Phase.FAILED.value]:
        # Job exists and is incomplete - resume it
        console.print(f"[dim]Job ID: {job_id}[/dim]")
        console.print(f"[yellow]⚠️  Job already exists. Resuming from {existing_state.get('current_phase')}...[/yellow]\n")
        initial_state = existing_state
        # Add key manager (can't be serialized, so recreate on load)
        initial_state["key_manager"] = KeyManager.from_env()
    else:
        # Create new job (fresh or overwriting completed/failed job)
        job_id = job_manager.create_job(title, context, length)
        console.print(f"[dim]Job ID: {job_id}[/dim]")
        console.print("")

        # Create initial state
        initial_state = {
            "job_id": job_id,
            "title": title,
            "context": context,
            "target_length": length,
            "current_phase": Phase.TOPIC_DISCOVERY.value,
            "current_section_index": 0,
            "section_drafts": {},
            "flags": {},
            "key_manager": KeyManager.from_env(),
        }

    try:
        # Build graph
        graph = build_blog_agent_graph()

        # Track state for progress display
        last_phase = None
        last_section_idx = 0
        result = initial_state

        # Stream through graph execution
        with Status("[bold blue]Starting...", console=console) as status:
            async for event in graph.astream(initial_state):
                # event is a dict with node name as key
                for node_name, node_output in event.items():
                    if node_output is None:
                        continue  # Skip nodes that returned None (e.g., skipped nodes)
                    result = {**result, **node_output}

                    current_phase = result.get("current_phase", "")
                    section_idx = result.get("current_section_index", 0)

                    # Update status based on phase changes
                    if current_phase != last_phase:
                        if last_phase:
                            # Show completion of previous phase
                            icon, _ = PHASE_MESSAGES.get(last_phase, ("✓", ""))
                            console.print(f"  [green]✓[/green] {last_phase.replace('_', ' ').title()} complete")

                        # Stop spinner for human review phase (needs interactive input)
                        if current_phase == Phase.REVIEWING.value:
                            status.stop()
                        else:
                            # Update spinner for new phase
                            status.update(_get_phase_message(current_phase))
                        last_phase = current_phase

                    # Special handling for writing phase - show section progress
                    elif current_phase == "writing" and section_idx != last_section_idx:
                        plan = result.get("plan", {})
                        sections = [s for s in plan.get("sections", []) if not s.get("optional")]
                        total = len(sections)

                        if section_idx > 0 and section_idx <= total:
                            prev_section = sections[section_idx - 1] if section_idx > 0 else None
                            if prev_section:
                                section_title = prev_section.get("title") or prev_section.get("role", "Section")
                                console.print(f"    [green]✓[/green] {section_title}")

                        if section_idx < total:
                            next_section = sections[section_idx]
                            section_title = next_section.get("title") or next_section.get("role", "Section")
                            status.update(f"✍️  Writing section {section_idx + 1}/{total}: {section_title}...")

                        last_section_idx = section_idx

                    # Check for failure
                    if current_phase == Phase.FAILED.value:
                        status.stop()
                        console.print(f"\n[bold red]❌ Generation failed:[/bold red]")
                        console.print(f"   {result.get('error_message', 'Unknown error')}")
                        sys.exit(1)

            # Final phase completion message
            if last_phase and last_phase != Phase.FAILED.value:
                console.print(f"  [green]✓[/green] {last_phase.replace('_', ' ').title()} complete")

        # Success
        job_dir = job_manager.get_job_dir(job_id)
        final_path = job_dir / "final.md"

        console.print(f"\n[bold green]✅ Blog generated successfully![/bold green]")
        console.print(f"[dim]Output:[/dim] {final_path}")

        # Show stats
        metadata = result.get("metadata", {})
        if metadata:
            console.print(f"\n[bold]Stats:[/bold]")
            console.print(f"  📝 Words: {metadata.get('word_count', 'N/A')}")
            console.print(f"  ⏱️  Reading time: {metadata.get('reading_time_minutes', 'N/A')} min")
            console.print(f"  📑 Sections: {metadata.get('section_count', 'N/A')}")

        # Show execution metrics
        display_metrics_summary(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted. Job saved for resume.[/yellow]")
        console.print(f"Resume with: [bold]python -m src.agent resume {job_id}[/bold]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("job_id")
def resume(job_id: str):
    """Resume an interrupted job."""
    asyncio.run(_run_resume(job_id))


async def _run_resume(job_id: str):
    """Resume a job from saved state with real-time progress updates."""
    job_manager = JobManager()

    # Load state
    state = job_manager.load_state(job_id)
    if state is None:
        console.print(f"[bold red]❌ Job not found:[/bold red] {job_id}")
        sys.exit(1)

    # Add key manager (can't be serialized, so recreate on load)
    state["key_manager"] = KeyManager.from_env()

    current_phase = state.get("current_phase", "")
    console.print(f"\n[bold blue]🔄 Resuming Job[/bold blue]")
    console.print(f"[dim]Job ID:[/dim] {job_id}")
    console.print(f"[dim]Current phase:[/dim] {current_phase}\n")

    if current_phase == Phase.DONE.value:
        console.print("[green]✅ Job already complete.[/green]")
        job_dir = job_manager.get_job_dir(job_id)
        console.print(f"[dim]Output:[/dim] {job_dir / 'final.md'}")
        return

    if current_phase == Phase.FAILED.value:
        console.print("[yellow]⚠️  Job previously failed. Restarting from beginning.[/yellow]")
        state["current_phase"] = Phase.TOPIC_DISCOVERY.value
        state["current_section_index"] = 0

    # Handle REVIEWING phase directly (skip graph, just run human review)
    if current_phase == Phase.REVIEWING.value:
        from .nodes import human_review_node

        result = await human_review_node(state)
        state.update(result)

        if state.get("human_review_decision") == "approve":
            job_manager.save_state(job_id, {"current_phase": Phase.DONE.value})
            job_dir = job_manager.get_job_dir(job_id)
            console.print(f"\n[bold green]✅ Blog finalized![/bold green]")
            console.print(f"[dim]Output:[/dim] {job_dir / 'final.md'}")

            # Display metrics if available
            display_metrics_summary(state)
        else:
            console.print("\n[yellow]Job cancelled.[/yellow]")
        return

    try:
        # Build graph
        graph = build_blog_agent_graph()

        # Track state for progress display
        last_phase = None
        last_section_idx = state.get("current_section_index", 0)
        result = state

        # Stream through graph execution
        with Status("[bold blue]Resuming...", console=console) as status:
            async for event in graph.astream(state):
                # event is a dict with node name as key
                for node_name, node_output in event.items():
                    if node_output is None:
                        continue  # Skip nodes that returned None (e.g., skipped nodes)
                    result = {**result, **node_output}

                    current_phase = result.get("current_phase", "")
                    section_idx = result.get("current_section_index", 0)

                    # Update status based on phase changes
                    if current_phase != last_phase:
                        if last_phase:
                            console.print(f"  [green]✓[/green] {last_phase.replace('_', ' ').title()} complete")

                        # Stop spinner for human review phase (needs interactive input)
                        if current_phase == Phase.REVIEWING.value:
                            status.stop()
                        else:
                            # Update spinner for new phase
                            status.update(_get_phase_message(current_phase))
                        last_phase = current_phase

                    # Special handling for writing phase
                    elif current_phase == "writing" and section_idx != last_section_idx:
                        plan = result.get("plan", {})
                        sections = [s for s in plan.get("sections", []) if not s.get("optional")]
                        total = len(sections)

                        if section_idx > 0 and section_idx <= total:
                            prev_section = sections[section_idx - 1] if section_idx > 0 else None
                            if prev_section:
                                section_title = prev_section.get("title") or prev_section.get("role", "Section")
                                console.print(f"    [green]✓[/green] {section_title}")

                        if section_idx < total:
                            next_section = sections[section_idx]
                            section_title = next_section.get("title") or next_section.get("role", "Section")
                            status.update(f"✍️  Writing section {section_idx + 1}/{total}: {section_title}...")

                        last_section_idx = section_idx

                    # Check for failure
                    if current_phase == Phase.FAILED.value:
                        status.stop()
                        console.print(f"\n[bold red]❌ Generation failed:[/bold red]")
                        console.print(f"   {result.get('error_message', 'Unknown error')}")
                        sys.exit(1)

            # Final phase completion message
            if last_phase and last_phase != Phase.FAILED.value:
                console.print(f"  [green]✓[/green] {last_phase.replace('_', ' ').title()} complete")

        job_dir = job_manager.get_job_dir(job_id)
        console.print(f"\n[bold green]✅ Blog generated successfully![/bold green]")
        console.print(f"[dim]Output:[/dim] {job_dir / 'final.md'}")

        # Show stats
        metadata = result.get("metadata", {})
        if metadata:
            console.print(f"\n[bold]Stats:[/bold]")
            console.print(f"  📝 Words: {metadata.get('word_count', 'N/A')}")
            console.print(f"  ⏱️  Reading time: {metadata.get('reading_time_minutes', 'N/A')} min")
            console.print(f"  📑 Sections: {metadata.get('section_count', 'N/A')}")

        # Show execution metrics
        display_metrics_summary(result)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted. Job saved for resume.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.option(
    "--status",
    type=click.Choice(["complete", "incomplete"]),
    default=None,
    help="Filter by status",
)
def jobs(status: str | None):
    """List all jobs."""
    job_manager = JobManager()
    job_list = job_manager.list_jobs(status)

    if not job_list:
        console.print("[dim]No jobs found.[/dim]")
        return

    console.print(f"\n[bold]Jobs ({len(job_list)} total)[/bold]\n")

    for job in job_list:
        status_icon = "[green]✓[/green]" if job.get("complete") else "[yellow]○[/yellow]"
        console.print(f"  {status_icon} {job['job_id']}")
        console.print(f"      [dim]{job.get('title', 'Untitled')}[/dim]")
        console.print(f"      [dim]Phase: {job.get('phase', 'unknown')}[/dim]")


@cli.command()
@click.argument("job_id")
def show(job_id: str):
    """Show details of a specific job."""
    job_manager = JobManager()
    state = job_manager.load_state(job_id)

    if state is None:
        console.print(f"[bold red]Job not found:[/bold red] {job_id}")
        sys.exit(1)

    console.print(f"\n[bold]Job: {job_id}[/bold]\n")
    console.print(f"Title: {state.get('title', 'N/A')}")
    console.print(f"Phase: {state.get('current_phase', 'N/A')}")
    console.print(f"Target length: {state.get('target_length', 'N/A')}")

    if state.get("plan"):
        sections = state["plan"].get("sections", [])
        console.print(f"Sections: {len(sections)}")

    if state.get("section_drafts"):
        console.print(f"Drafts written: {len(state['section_drafts'])}")

    job_dir = job_manager.get_job_dir(job_id)
    if (job_dir / "final.md").exists():
        console.print(f"\n[green]Output:[/green] {job_dir / 'final.md'}")


if __name__ == "__main__":
    cli()
