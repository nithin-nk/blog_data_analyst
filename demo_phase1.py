#!/usr/bin/env python
"""
Demo script for Phase 1 components.

Demonstrates the foundation components working together.
"""

from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config.settings import get_settings
from src.parsers.yaml_parser import YAMLParser
from src.utils.logger import setup_logger
from src.utils.file_handler import FileHandler


console = Console()
logger = setup_logger(__name__)


def main():
    """Demonstrate Phase 1 functionality."""
    console.print(Panel.fit(
        "[bold cyan]Blog Data Analyst - Phase 1 Demo[/bold cyan]\n"
        "Foundation Components",
        border_style="cyan"
    ))
    
    # 1. Configuration
    console.print("\n[bold]1. Configuration System[/bold]")
    settings = get_settings()
    
    config_table = Table(show_header=True, header_style="bold magenta")
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    
    config_table.add_row("Environment", settings.environment)
    config_table.add_row("Log Level", settings.log_level)
    config_table.add_row("Max Retries", str(settings.max_retries))
    config_table.add_row("Quality Threshold", str(settings.quality_threshold))
    config_table.add_row("Max Refinement Iterations", str(settings.max_refinement_iterations))
    
    console.print(config_table)
    
    # 2. YAML Parser
    console.print("\n[bold]2. YAML Parser[/bold]")
    
    example_file = Path("inputs/example_input.yaml")
    if example_file.exists():
        blog_input = YAMLParser.parse_file(example_file)
        
        console.print(f"[cyan]Topic:[/cyan] {blog_input.topic}")
        console.print(f"[cyan]Outline Items:[/cyan] {len(blog_input.outline)}")
        console.print(f"[cyan]Metadata:[/cyan] {blog_input.metadata}")
        
        # Show outline with markers
        outline_table = Table(show_header=True, header_style="bold yellow")
        outline_table.add_column("#", style="dim", width=3)
        outline_table.add_column("Section", style="white")
        outline_table.add_column("Markers", style="magenta")
        
        for i, item in enumerate(blog_input.outline_items, 1):
            markers = []
            if item.requires_code:
                markers.append("📝 Code")
            if item.requires_mermaid:
                markers.append("📊 Mermaid")
            marker_str = ", ".join(markers) if markers else "-"
            
            outline_table.add_row(
                str(i),
                item.clean_text[:60] + "..." if len(item.clean_text) > 60 else item.clean_text,
                marker_str
            )
        
        console.print(outline_table)
        
        # Statistics
        code_count = sum(1 for item in blog_input.outline_items if item.requires_code)
        mermaid_count = sum(1 for item in blog_input.outline_items if item.requires_mermaid)
        
        console.print(f"\n[green]✓[/green] Detected {code_count} code section(s)")
        console.print(f"[green]✓[/green] Detected {mermaid_count} Mermaid diagram(s)")
    else:
        console.print("[yellow]No example file found[/yellow]")
    
    # 3. File Handler
    console.print("\n[bold]3. File Handler[/bold]")
    
    # Create a test blog structure
    temp_dir = Path("outputs") / "demo_test"
    paths = FileHandler.create_blog_structure("demo-blog", temp_dir)
    
    console.print("[green]✓[/green] Created blog directory structure:")
    for key, path in paths.items():
        if key != "blog_dir":
            try:
                rel_path = path.relative_to(Path.cwd())
            except ValueError:
                rel_path = path
            console.print(f"  • {key}: {rel_path}")
    
    # 4. Logger
    console.print("\n[bold]4. Logger System[/bold]")
    console.print("[green]✓[/green] Colored, structured logging enabled")
    console.print("[green]✓[/green] Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL")
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    
    # Summary
    console.print(Panel.fit(
        "[bold green]✅ Phase 1 Complete[/bold green]\n\n"
        "Foundation components are working:\n"
        "• Configuration system with environment variables\n"
        "• YAML parser with marker detection\n"
        "• File handler with safe I/O\n"
        "• Structured logging with colors\n\n"
        "[cyan]Ready for Phase 2: Research & Content Extraction[/cyan]",
        border_style="green",
        title="[bold]Summary[/bold]"
    ))


if __name__ == "__main__":
    main()
