"""
CLI module for the Blog Agent.

Usage:
    python -m src.agent start --title "..." --context "..."
    python -m src.agent resume <job_id>
    python -m src.agent jobs [--status complete|incomplete]
"""

import asyncio
import sys

import click
from rich.console import Console

from .graph import build_blog_agent_graph
from .state import JobManager, Phase

console = Console()


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
    """Execute the blog generation pipeline."""
    console.print(f"\n[bold blue]Blog Agent[/bold blue]")
    console.print(f"Title: {title}")
    console.print(f"Length: {length}\n")

    # Create job
    job_manager = JobManager()
    job_id = job_manager.create_job(title, context, length)

    console.print(f"[dim]Job ID: {job_id}[/dim]")
    console.print("[dim]Starting pipeline...[/dim]\n")

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
    }

    try:
        # Build and run graph
        graph = build_blog_agent_graph()
        result = await graph.ainvoke(initial_state)

        # Check result
        final_phase = result.get("current_phase", "")

        if final_phase == Phase.FAILED.value:
            console.print(f"\n[bold red]Generation failed:[/bold red]")
            console.print(f"  {result.get('error_message', 'Unknown error')}")
            sys.exit(1)

        # Success
        job_dir = job_manager.get_job_dir(job_id)
        final_path = job_dir / "final.md"

        console.print(f"\n[bold green]Blog generated successfully![/bold green]")
        console.print(f"Output: {final_path}")

        # Show stats
        metadata = result.get("metadata", {})
        if metadata:
            console.print(f"\n[dim]Stats:[/dim]")
            console.print(f"  Words: {metadata.get('word_count', 'N/A')}")
            console.print(f"  Reading time: {metadata.get('reading_time_minutes', 'N/A')} min")
            console.print(f"  Sections: {metadata.get('section_count', 'N/A')}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Job saved for resume.[/yellow]")
        console.print(f"Resume with: python -m src.agent resume {job_id}")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@cli.command()
@click.argument("job_id")
def resume(job_id: str):
    """Resume an interrupted job."""
    asyncio.run(_run_resume(job_id))


async def _run_resume(job_id: str):
    """Resume a job from saved state."""
    job_manager = JobManager()

    # Load state
    state = job_manager.load_state(job_id)
    if state is None:
        console.print(f"[bold red]Job not found:[/bold red] {job_id}")
        sys.exit(1)

    current_phase = state.get("current_phase", "")
    console.print(f"\n[bold blue]Resuming Job[/bold blue]")
    console.print(f"Job ID: {job_id}")
    console.print(f"Current phase: {current_phase}\n")

    if current_phase == Phase.DONE.value:
        console.print("[green]Job already complete.[/green]")
        job_dir = job_manager.get_job_dir(job_id)
        console.print(f"Output: {job_dir / 'final.md'}")
        return

    if current_phase == Phase.FAILED.value:
        console.print("[yellow]Job previously failed. Restarting from beginning.[/yellow]")
        state["current_phase"] = Phase.TOPIC_DISCOVERY.value
        state["current_section_index"] = 0

    try:
        # Build and run graph
        graph = build_blog_agent_graph()
        result = await graph.ainvoke(state)

        final_phase = result.get("current_phase", "")

        if final_phase == Phase.FAILED.value:
            console.print(f"\n[bold red]Generation failed:[/bold red]")
            console.print(f"  {result.get('error_message', 'Unknown error')}")
            sys.exit(1)

        job_dir = job_manager.get_job_dir(job_id)
        console.print(f"\n[bold green]Blog generated successfully![/bold green]")
        console.print(f"Output: {job_dir / 'final.md'}")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Job saved for resume.[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
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
