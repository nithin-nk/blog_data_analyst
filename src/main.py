"""
Main orchestrator for the Blog Data Analyst AI Agent.

This module coordinates the entire pipeline:
1. Parse YAML input
2. Research topics (Question Generation + Search)
3. Generate content
4. Optimize and refine
5. Generate media
6. Convert to HTML
7. Save outputs
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config.settings import get_settings
from src.planning.question_generator import QuestionGenerator
from src.planning.outline_generator import OutlineGenerator
from src.planning.outline_reviewer import OutlineReviewer
from src.research.search_agent import SearchAgent
from src.research.content_extractor import ContentExtractor
from src.utils.file_handler import FileHandler
from src.utils.logger import setup_logger


console = Console()
logger = setup_logger(__name__)


async def run_research_pipeline(
    topic: str,
    context: Optional[str],
    output_dir: Path,
    results_per_query: int = 3,
    rate_limit_delay: float = 1.5,
    extraction_concurrency: int = 5,
    extraction_timeout: int = 30,
) -> dict:
    """
    Run the research pipeline (Steps 2.1 + 2.2 + 2.3).
    
    Args:
        topic: Blog topic/title
        context: Optional additional context
        output_dir: Base output directory
        results_per_query: Number of search results per query
        rate_limit_delay: Delay between search requests
        extraction_concurrency: Number of concurrent content extractions
        extraction_timeout: Timeout for each extraction in seconds
        
    Returns:
        Dict with paths and results
    """
    # Create blog structure
    paths = FileHandler.create_blog_structure(topic, output_dir)
    
    console.print(Panel(
        f"[bold]Topic:[/bold] {topic}\n"
        f"[bold]Context:[/bold] {context or 'None'}\n"
        f"[bold]Output:[/bold] {paths['blog_dir']}",
        title="🔬 Research Pipeline",
        border_style="blue",
    ))
    
    # Step 2.1: Generate research questions
    with console.status("[bold cyan]Generating research questions...") as status:
        generator = QuestionGenerator(model_name="gemini-2.0-flash")
        questions_result = await generator.generate(topic, context)
        
        console.print(f"\n[green]✓[/green] Generated {len(questions_result.questions)} research questions")
        
        # Display questions
        table = Table(title="Research Questions", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Query", style="cyan")
        
        for i, q in enumerate(questions_result.questions, 1):
            table.add_row(str(i), q)
        
        console.print(table)
        console.print(f"[dim]Categories: {', '.join(questions_result.categories_covered)}[/dim]\n")
    
    # Save research questions
    FileHandler.save_research_questions(
        file_path=paths["research_questions"],
        topic=topic,
        context=context,
        questions=questions_result.questions,
        categories=questions_result.categories_covered,
    )
    console.print(f"[green]✓[/green] Saved questions to: {paths['research_questions']}\n")
    
    # Step 2.2: Search for each question
    console.print("[bold cyan]Searching DuckDuckGo for each query...[/bold cyan]")
    
    agent = SearchAgent(
        results_per_query=results_per_query,
        rate_limit_delay=rate_limit_delay,
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=len(questions_result.questions))
        
        search_results = await agent.search_multiple(
            questions_result.questions,
            deduplicate=True,
        )
        
        progress.update(task, completed=len(questions_result.questions))
    
    # Display search results summary
    console.print(f"\n[green]✓[/green] Search complete!")
    console.print(f"   Successful queries: {search_results.successful_queries}/{len(questions_result.questions)}")
    console.print(f"   Total results: {search_results.total_results}")
    console.print(f"   Unique URLs: {len(search_results.all_urls)}\n")
    
    # Display top URLs
    if search_results.all_urls:
        table = Table(title="Unique URLs Found", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("URL", style="blue")
        table.add_column("Found Via", style="dim")
        
        for i, url in enumerate(search_results.all_urls[:15], 1):  # Show top 15
            queries = search_results.url_to_queries.get(url, [])
            queries_str = queries[0][:40] + "..." if queries else ""
            if len(queries) > 1:
                queries_str += f" (+{len(queries)-1} more)"
            table.add_row(str(i), url[:80] + ("..." if len(url) > 80 else ""), queries_str)
        
        if len(search_results.all_urls) > 15:
            table.add_row("...", f"[dim]and {len(search_results.all_urls) - 15} more URLs[/dim]", "")
        
        console.print(table)
    
    # Convert search results to serializable format
    query_results_data = []
    for qr in search_results.queries:
        query_results_data.append({
            "query": qr.query,
            "success": qr.success,
            "error": qr.error,
            "results": [
                {"title": r.title, "url": r.url, "snippet": r.snippet}
                for r in qr.results
            ],
        })
    
    # Save search results
    FileHandler.save_search_results(
        file_path=paths["search_results"],
        topic=topic,
        query_results=query_results_data,
        all_urls=search_results.all_urls,
        url_to_queries=search_results.url_to_queries,
        stats={
            "total_queries": len(questions_result.questions),
            "successful_queries": search_results.successful_queries,
            "failed_queries": search_results.failed_queries,
            "total_results": search_results.total_results,
        },
    )
    console.print(f"\n[green]✓[/green] Saved search results to: {paths['search_results']}")

    # Step 2.3: Extract content from URLs
    if search_results.all_urls:
        console.print(f"\n[bold cyan]Extracting content from {len(search_results.all_urls)} URLs...[/bold cyan]")
        console.print(f"[dim](concurrency={extraction_concurrency}, timeout={extraction_timeout}s)[/dim]")

        # Build url_to_snippet mapping (keep longest snippet for each URL)
        url_to_snippet: dict[str, str] = {}
        for query_result in search_results.queries:
            for result in query_result.results:
                current_snippet = url_to_snippet.get(result.url, "")
                if len(result.snippet) > len(current_snippet):
                    url_to_snippet[result.url] = result.snippet

        extractor = ContentExtractor(
            concurrency_limit=extraction_concurrency,
            timeout=extraction_timeout,
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Extracting content from {len(search_results.all_urls)} URLs...",
                total=None,
            )

            extracted_content = await extractor.extract_multiple(
                urls=search_results.all_urls,
                url_to_queries=search_results.url_to_queries,
                url_to_snippet=url_to_snippet,
                topic=topic,
            )

            progress.update(task, completed=True)

        # Display extraction results summary
        stats = extracted_content.statistics
        console.print(f"\n[green]✓[/green] Content extraction complete!")
        console.print(f"   Successful: {stats.get('successful', 0)}/{stats.get('total_urls', 0)}")
        console.print(f"   Failed: {stats.get('failed', 0)}")

        # Display extracted content summary
        if extracted_content.contents:
            table = Table(title="Extracted Content Summary", show_lines=True)
            table.add_column("#", style="dim", width=3)
            table.add_column("URL", style="blue", max_width=50)
            table.add_column("Title", style="cyan", max_width=30)
            table.add_column("Markdown", style="dim", width=10)
            table.add_column("Headings", style="dim", width=8)
            table.add_column("Code", style="dim", width=6)
            table.add_column("Status", width=8)

            for i, content in enumerate(extracted_content.contents[:15], 1):
                url_short = content.url[:47] + "..." if len(content.url) > 50 else content.url
                title_short = content.title[:27] + "..." if len(content.title) > 30 else content.title
                status = "[green]✓[/green]" if content.success else f"[red]✗[/red]"

                table.add_row(
                    str(i),
                    url_short,
                    title_short or "[dim]N/A[/dim]",
                    f"{len(content.markdown):,}",
                    str(len(content.headings)),
                    str(len(content.code_blocks)),
                    status,
                )

            if len(extracted_content.contents) > 15:
                table.add_row(
                    "...",
                    f"[dim]and {len(extracted_content.contents) - 15} more URLs[/dim]",
                    "", "", "", "", ""
                )

            console.print(table)

        # Convert to serializable format and save
        contents_data = []
        for content in extracted_content.contents:
            contents_data.append({
                "url": content.url,
                "title": content.title,
                "snippet": content.snippet,
                "markdown": content.markdown,
                "headings": content.headings,
                "code_blocks": content.code_blocks,
                "success": content.success,
                "error": content.error,
                "extracted_at": content.extracted_at,
                "source_queries": content.source_queries,
            })

        FileHandler.save_extracted_content(
            file_path=paths["extracted_content"],
            topic=topic,
            contents=contents_data,
            statistics=stats,
        )
        console.print(f"\n[green]✓[/green] Saved extracted content to: {paths['extracted_content']}")
    else:
        extracted_content = None
        console.print("\n[yellow]⚠[/yellow] No URLs to extract content from")

    # Step 3.2 + 3.3: Generate and Review Outline
    if extracted_content and extracted_content.contents:
        console.print("\n[bold cyan]Generating and reviewing blog outline...[/bold cyan]")
        settings = get_settings()
        
        outline_generator = OutlineGenerator(model_name="gemini-2.0-flash")
        outline_reviewer = OutlineReviewer(model_name=settings.outline_reviewer_model)
        
        # Generate initial outline
        console.print("[dim]Generating initial outline...[/dim]")
        outline = await outline_generator.generate(topic, extracted_content)
        reviews = []
        
        # Review and iterate until quality threshold met
        max_iterations = settings.max_outline_iterations
        quality_threshold = settings.outline_quality_threshold
        
        # Track all iterations to fallback to best score if needed
        iteration_history = []  # List of (score, outline, review)

        for iteration in range(1, max_iterations + 1):
            console.print(f"\n[bold cyan]━━━ Iteration {iteration}/{max_iterations} ━━━[/bold cyan]")
            
            # Review current outline
            console.print("[dim]Reviewing outline quality...[/dim]")
            review = await outline_reviewer.review(outline)
            reviews.append(review)
            
            # Store iteration data
            iteration_history.append((review.score, outline, review))
            
            # Display review feedback
            outline_reviewer.display_review(review, iteration, console)
            
            if review.score >= quality_threshold:
                console.print(f"[green]✓ Outline approved! Score: {review.score:.1f}/{quality_threshold}[/green]")
                break
            
            # If not last iteration, regenerate with feedback
            if iteration < max_iterations:
                console.print(
                    f"[yellow]Score {review.score:.1f} < {quality_threshold}. Regenerating with feedback...[/yellow]"
                )
                outline = await outline_reviewer._regenerate_with_feedback(
                    topic, extracted_content, outline_generator, review
                )
            else:
                console.print(
                    f"[yellow]⚠ Max iterations reached. Final score: {review.score:.1f}/{quality_threshold}[/yellow]"
                )
                
                # Fallback logic: Select best iteration
                best_iteration = max(iteration_history, key=lambda x: x[0])
                best_score, best_outline, best_review = best_iteration
                
                if best_score > review.score:
                    console.print(f"\n[bold yellow]Reverting to best iteration (Score: {best_score:.1f})...[/bold yellow]")
                    outline = best_outline
                    # We might want to append the best review again or just rely on the history
                    # But for the final score reporting, we should probably use the best score.
                    # The reviews list already contains all reviews.
                    # Let's ensure the final reported score matches the selected outline.
                    # We can't easily "fix" the reviews list to make the last one the best one without confusing history,
                    # but we can ensure the return value uses the best outline.
                    
                    # Update reviews list to reflect the final decision? 
                    # Or just proceed with the best outline. 
                    # The `reviews[-1]` is used later for score reporting. 
                    # Let's append the best review again to `reviews` so it looks like the final state?
                    # Or just rely on `iteration_history`.
                    
                    # Simplest approach: Just set outline to best_outline. 
                    # The `reviews` list is for history. 
                    # But `reviews[-1]` is used for final score display.
                    # So let's append the best review to `reviews` if it's not the last one.
                    if best_review != reviews[-1]:
                         reviews.append(best_review)
                         console.print(f"[dim]Restored review from best iteration.[/dim]")
        
        # Save all reviews to YAML
        await outline_reviewer.save_reviews(reviews, paths["outline_reviews"])
        console.print(f"[green]✓[/green] Saved {len(reviews)} review(s) to: {paths['outline_reviews']}")
        
        # Display final summary
        console.print(f"\n[green]✓[/green] Outline finalized after {len(reviews)} iteration(s)")
        console.print(f"   Final Score: {reviews[-1].score:.1f}/{settings.outline_quality_threshold}")
        console.print(f"   Target Audience: {outline.metadata.target_audience}")
        console.print(f"   Difficulty: {outline.metadata.difficulty}")
        
        # Display outline structure
        table = Table(title="Final Blog Outline", show_lines=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("Section", style="cyan")
        table.add_column("Summary", style="dim")
        
        for i, section in enumerate(outline.sections, 1):
            table.add_row(str(i), section.heading, section.summary[:100] + "...")
            
        console.print(table)
        
        # Save outline to research directory
        FileHandler.save_outline(
            file_path=paths["outline"],
            outline_data=outline.model_dump(),
        )
        console.print(f"[green]✓[/green] Saved outline to: {paths['outline']}")
        
        # Save approved outline to inputs directory for human review
        approved_path = FileHandler.save_approved_outline(
            topic=topic,
            outline_data=outline.model_dump(),
            input_dir=settings.input_dir,
        )
        console.print(f"[green]✓[/green] Saved approved outline to: {approved_path}")
        
    else:
        outline = None
        console.print("\n[yellow]⚠[/yellow] Skipping outline generation (no content)")

    return {
        "paths": paths,
        "questions": questions_result,
        "search_results": search_results,
        "extracted_content": extracted_content,
        "outline": outline,
        "final_score": reviews[-1].score if (extracted_content and extracted_content.contents and outline) else None,
    }


@click.group()
def cli():
    """Blog Data Analyst AI Agent - Generate high-quality blog posts."""
    pass


@cli.command()
@click.option(
    "--topic",
    "-t",
    required=True,
    help="Blog topic/title to research",
)
@click.option(
    "--context",
    "-c",
    default=None,
    help="Additional context or constraints for the research",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: from settings)",
)
@click.option(
    "--results-per-query",
    type=int,
    default=3,
    help="Number of search results per query (default: 3)",
)
@click.option(
    "--rate-limit",
    type=float,
    default=1.5,
    help="Delay between search requests in seconds (default: 1.5)",
)
@click.option(
    "--extraction-concurrency",
    type=int,
    default=5,
    help="Number of concurrent content extractions (default: 5)",
)
@click.option(
    "--extraction-timeout",
    type=int,
    default=30,
    help="Timeout for each extraction in seconds (default: 30)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def research(
    topic: str,
    context: Optional[str],
    output_dir: Optional[Path],
    results_per_query: int,
    rate_limit: float,
    extraction_concurrency: int,
    extraction_timeout: int,
    verbose: bool,
) -> None:
    """
    Run research pipeline (Steps 2.1 + 2.2 + 2.3).
    
    Generates research questions, searches DuckDuckGo for relevant URLs,
    and extracts content from discovered pages.
    Results are saved to outputs/<topic>/research/
    """
    settings = get_settings()
    
    if verbose:
        logger.setLevel("DEBUG")
    
    if output_dir is None:
        output_dir = settings.output_dir
    
    console.print("\n[bold blue]🚀 Blog Data Analyst - Research Pipeline[/bold blue]\n")
    
    try:
        result = asyncio.run(run_research_pipeline(
            topic=topic,
            context=context,
            output_dir=output_dir,
            results_per_query=results_per_query,
            rate_limit_delay=rate_limit,
            extraction_concurrency=extraction_concurrency,
            extraction_timeout=extraction_timeout,
        ))
        
        # Build success message
        extracted_info = ""
        if result.get("extracted_content"):
            stats = result["extracted_content"].statistics
            extracted_info = f"Extracted Content: {stats.get('successful', 0)}/{stats.get('total_urls', 0)} successful\n"
        
        outline_info = ""
        next_command = ""
        if result.get("outline"):
            outline_info = f"Outline: {len(result['outline'].sections)} sections (Score: {result.get('final_score', 'N/A')}/10)\n"
            
            # Generate next command
            settings = get_settings()
            topic_slug = FileHandler.slugify(topic)
            approved_file = settings.input_dir / f"{topic_slug}.yaml"
            next_command = f"\n[bold cyan]Next Steps:[/bold cyan]\n"
            next_command += f"1. Review the outline at: [yellow]{approved_file}[/yellow]\n"
            next_command += f"2. Edit if needed, then run:\n"
            next_command += f"   [green]python -m src.main generate --input {approved_file}[/green]"
        
        console.print(Panel(
            f"[green]Research complete![/green]\n\n"
            f"Questions: {len(result['questions'].questions)}\n"
            f"Unique URLs: {len(result['search_results'].all_urls)}\n"
            f"{extracted_info}"
            f"{outline_info}\n"
            f"Output directory: {result['paths']['blog_dir']}\n"
            f"{next_command}",
            title="✅ Success",
            border_style="green",
        ))
        
    except Exception as e:
        logger.error(f"Research pipeline failed: {e}", exc_info=True)
        console.print(f"\n[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


@cli.command()
@click.option(
    "--input",
    "-i",
    "input_file",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the YAML input file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for generated content",
)
@click.option(
    "--max-iterations",
    type=int,
    default=3,
    help="Maximum refinement iterations",
)
@click.option(
    "--skip-image",
    is_flag=True,
    help="Skip image generation",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Run without making API calls (testing mode)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging",
)
def generate(
    input_file: Path,
    output_dir: Optional[Path],
    max_iterations: int,
    skip_image: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """
    Generate blog post from YAML specification.
    
    This is the full pipeline that takes an approved outline and generates
    the complete blog post with content, code, diagrams, and images.
    """
    settings = get_settings()
    
    if verbose:
        logger.setLevel("DEBUG")
    
    console.print("[bold blue]🚀 Blog Data Analyst AI Agent[/bold blue]")
    console.print(f"Input: {input_file}")
    
    if dry_run:
        console.print("[yellow]⚠️  Running in DRY RUN mode[/yellow]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Processing...", total=10)
            
            # Step 1: Parse YAML
            progress.update(task, advance=1, description="[cyan]Parsing YAML input...")
            logger.info(f"Parsing input file: {input_file}")
            
            outline_data = FileHandler.read_yaml(input_file)
            topic = outline_data.get("topic")
            if not topic:
                raise ValueError("Input YAML must contain a 'topic' field")
            
            # Setup paths
            if output_dir is None:
                output_dir = settings.output_dir
            
            paths = FileHandler.create_blog_structure(topic, output_dir)
            
            # Step 2: Research (Skipped in generate mode, assuming it exists)
            progress.update(task, advance=1, description="[cyan]Checking research data...")
            research_path = paths["extracted_content"]
            if not research_path.exists():
                logger.warning(f"Research data not found at {research_path}. Content generation may be poor.")
                # In a real scenario, we might want to fail or trigger research here.
                # For now, we'll proceed but the generator might fail if it strictly needs it.
                # Actually, the generator needs it.
                if not dry_run:
                     raise FileNotFoundError(f"Research data not found at {research_path}. Please run 'research' command first.")
            
            # Step 3: Generate content
            progress.update(task, advance=1, description="[cyan]Generating content...")
            logger.info("Generating blog content")
            
            if not dry_run:
                from src.generation.content_generator import ContentGenerator
                content_generator = ContentGenerator()
                
                def update_progress(msg):
                    progress.update(task, description=f"[cyan]{msg}")
                    console.print(msg)  # Also print directly to console
                
                draft_content = content_generator.generate_blog_post(input_file, research_path, progress_callback=update_progress)
                
                # Save draft
                FileHandler.write_file(paths["draft"], draft_content)
                console.print(f"[green]✓[/green] Draft content saved to: {paths['draft']}")
            else:
                draft_content = "# Dry Run Content"
                console.print("[yellow]Dry run: Skipping content generation[/yellow]")
            
            # Step 4: Generate diagrams
            progress.update(task, advance=1, description="[cyan]Generating mermaid diagrams...")
            
            diagrams_path = paths["blog_dir"] / "diagrams.yaml"
            
            if not dry_run and draft_content:
                from src.generation.diagram_generator import DiagramGenerator
                
                def diagram_progress(msg):
                    console.print(msg)
                
                diagram_generator = DiagramGenerator()
                diagrams = diagram_generator.generate_all_diagrams(
                    title=topic,
                    content=draft_content,
                    progress_callback=diagram_progress
                )
                
                if diagrams:
                    # Save diagrams to YAML
                    diagram_generator.save_diagrams(diagrams, diagrams_path)
                    console.print(f"[green]✓[/green] Generated {len(diagrams)} diagrams, saved to: {diagrams_path}")
                else:
                    console.print("[yellow]No diagrams generated[/yellow]")
            else:
                console.print("[yellow]Dry run: Skipping diagram generation[/yellow]")
            
            # Step 4.5: Generate blog cover image
            progress.update(task, advance=0.5, description="[cyan]Generating blog cover image...")
            
            if not dry_run and draft_content and not skip_image:
                from src.media.blog_image_generator import BlogImageGenerator
                
                def image_progress(msg):
                    console.print(msg)
                
                try:
                    blog_image_generator = BlogImageGenerator()
                    blog_image = blog_image_generator.generate_blog_image(
                        title=topic,
                        content=draft_content,
                        progress_callback=image_progress
                    )
                    
                    # Save to diagrams.yaml
                    blog_image_generator.save_to_diagrams_yaml(blog_image, diagrams_path)
                    console.print(f"[green]✓[/green] Blog cover image generated and saved to: {diagrams_path}")
                    
                except Exception as e:
                    logger.warning(f"Blog image generation failed: {e}")
                    console.print(f"[yellow]⚠ Blog image generation failed: {e}[/yellow]")
            elif skip_image:
                console.print("[yellow]Skipping blog image generation (--skip-image flag)[/yellow]")
            else:
                console.print("[yellow]Dry run: Skipping blog image generation[/yellow]")
            
            # Step 5: Combine sections
            progress.update(task, advance=1, description="[cyan]Combining sections...")
            # TODO: Implement section combining
            
            # Step 6: Generate title/tags/meta
            progress.update(task, advance=1, description="[cyan]Generating metadata...")
            # TODO: Implement metadata generation
            
            # Step 7: SEO optimization
            progress.update(task, advance=1, description="[cyan]Optimizing for SEO...")
            # TODO: Implement SEO optimization
            
            # Step 8: Quality check & refinement
            progress.update(task, advance=1, description="[cyan]Quality checking...")
            # TODO: Implement quality check and refinement loop
            
            # Step 9: Image generation handled in Step 4.5
            if not skip_image:
                progress.update(task, advance=1, description="[cyan]Blog image already generated...")
            else:
                progress.update(task, advance=1)
            
            # Step 10: Convert to HTML & save
            progress.update(task, advance=1, description="[cyan]Converting to HTML...")
            # TODO: Implement HTML conversion and saving
            
        console.print("[bold green]✅ Blog generation complete![/bold green]")
        console.print(f"Draft saved to: {paths['draft']}")
        
    except Exception as e:
        logger.error(f"Error during blog generation: {e}", exc_info=True)
        console.print(f"[bold red]❌ Error: {e}[/bold red]")
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
