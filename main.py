"""
🎨 Vision-X - CLI Interface
Command line interface for the image analysis tool.
"""

import asyncio
import argparse
from pathlib import Path
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeRemainingColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.layout import Layout
from rich.tree import Tree
from rich.box import ROUNDED, DOUBLE
from rich.style import Style
from rich.text import Text
from rich.padding import Padding
import json
from datetime import datetime
from image_analyzer import analyze_image, logger
import aiofiles
from asyncio import Lock

console = Console()
progress_lock = Lock()


def save_results(results: dict, output_file: str, format: str = "json") -> None:
    """Save the analysis results to a file."""
    output_data = {
        "analysis_timestamp": datetime.now().isoformat(),
        "total_images": len(results),
        "successful_analyses": sum(1 for r in results.values() if not r.get('error') and len(r) > 1),
        "failed_analyses": sum(1 for r in results.values() if r.get('error') or len(r) <= 1),
        "results": results,
    }

    try:
        if format == "json":
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
        else:  # format == "txt"
            with open(output_file, "w", encoding="utf-8") as f:
                for img_name, analysis in results.items():
                    f.write(f"=== {img_name} ===\n")
                    if "error" in analysis:
                        f.write(f"Error: {analysis['error']}\n")
                    else:
                        f.write(f"Summary: {analysis['summary']}\n")
                        f.write(f"Scene: {analysis['scene']}\n")
                        f.write(
                            f"Objects: {', '.join(str(obj) for obj in analysis['objects'])}\n"
                        )
                        f.write("\n")
        logger.info(f"Results saved successfully to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}", exc_info=True)


async def load_progress(progress_file: str = "analysis_progress.json") -> dict:
    """Load previous analysis progress if it exists."""
    try:
        async with aiofiles.open(progress_file, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    except FileNotFoundError:
        return {"processed_files": [], "results": {}}
    except Exception as e:
        logger.error(f"Error loading progress: {str(e)}", exc_info=True)
        return {"processed_files": [], "results": {}}


async def save_progress(
    results: dict, processed_files: list, progress_file: str = "analysis_progress.json"
):
    """Save current analysis progress with file locking."""
    progress_data = {
        "timestamp": datetime.now().isoformat(),
        "processed_files": processed_files,
        "results": results,
    }
    async with progress_lock:  # Ensure only one save operation at a time
        try:
            async with aiofiles.open(progress_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(progress_data, indent=4, ensure_ascii=False))
            logger.info("Progress saved successfully")
        except Exception as e:
            logger.error(f"Failed to save progress: {str(e)}", exc_info=True)


async def process_single_file(image_path: Path, args) -> dict:
    """Process a single image file."""
    console.print(f"Processing image: {image_path}")
    image_desc, analysis_meta = await analyze_image(image_path)
    
    result = {
        image_path.name: {
            **(image_desc.model_dump() if image_desc else {}),
            **analysis_meta.model_dump(),
        }
    }
    
    if args.output:
        save_results(result, args.output, args.format)
    
    return result


def create_color_swatch(color_name: str, hex_code: str, prominence: float) -> Text:
    """Create a colored swatch with the color name and prominence."""
    swatch = Text("██████", style=f"rgb({hex_code.lstrip('#')})")
    return Text.assemble(
        swatch, " ", 
        color_name, " ", 
        f"({prominence:.1%})", 
        style=f"bold {color_name}" if color_name in {"red", "blue", "green", "yellow", "white", "black", "pink"} else ""
    )


def create_attributes_tree(attributes: list) -> str:
    """Create a formatted string view of object attributes."""
    if not attributes:
        return "No attributes found"
    
    # Create a simple indented list instead of using the Tree widget
    lines = []
    for attr in attributes:
        confidence_str = f"[green]{attr['confidence']:.1%}[/green]" if attr['confidence'] > 0.7 else f"[yellow]{attr['confidence']:.1%}[/yellow]"
        lines.append(f"  • [bold]{attr['name']}[/bold]: {attr['value']} ({confidence_str})")
    
    return "\n".join(lines)


async def display_image_analysis(file_path: str, full: bool = False, short: bool = False):
    """Display detailed analysis for a specific image."""
    progress_data = await load_progress()
    results = progress_data.get("results", {})
    
    image_name = Path(file_path).name
    if image_name not in results:
        console.print(f"[red]No analysis data found for: {image_name}[/red]")
        return

    analysis = results[image_name]
    
    if short:
        # Short display mode
        summary = Panel(
            f"[cyan]{analysis.get('summary', 'No summary available')}[/cyan]\n\n"
            f"[yellow]Tags:[/yellow] {', '.join(analysis.get('tags', []))}\n"
            f"[yellow]Scene:[/yellow] {analysis.get('scene', 'Unknown')}\n"
            f"[yellow]Quality:[/yellow] {analysis.get('image_quality', 'Unknown')}",
            title="🖼️ Image Analysis Summary",
            border_style="blue"
        )
        console.print(summary)
        return

    # Create layout for full or normal display
    layout = Layout()
    layout.split_column(
        Layout(name="header"),
        Layout(name="main", ratio=3),
        Layout(name="footer")
    )

    # Header with summary and basic info
    header_text = Text.assemble(
        ("🖼️ Image Analysis Result\n", "bold magenta"),
        ("\nFile: ", "bold"), image_name,
        ("\nAnalysis Duration: ", "bold"), f"{analysis.get('analysis_duration', 0):.2f}s",
        ("\nTimestamp: ", "bold"), analysis.get('timestamp', 'Unknown')
    )
    layout["header"].update(Panel(header_text, border_style="blue"))

    # Main content layout
    main_layout = Layout()
    main_layout.split_row(
        Layout(name="left", ratio=2),
        Layout(name="right", ratio=3)
    )

    # Left side - Summary and basic info
    left_panel_content = [
        Panel(
            f"[cyan]{analysis.get('summary', 'No summary available')}[/cyan]",
            title="Summary",
            border_style="blue",
            box=ROUNDED
        ),
        Panel(
            "\n".join([
                f"[bold]Scene:[/bold] {analysis.get('scene', 'Unknown')}",
                f"[bold]Setting:[/bold] {analysis.get('setting', 'Unknown')}",
                f"[bold]Time:[/bold] {analysis.get('time_of_day', 'Unknown')}",
                f"[bold]Quality:[/bold] {analysis.get('image_quality', 'Unknown')}"
            ]),
            title="Scene Information",
            border_style="green",
            box=ROUNDED
        )
    ]

    # Add character details if present
    if analysis.get('character_details'):
        char_details = analysis['character_details']
        char_panel = Panel(
            "\n".join([
                f"[bold]{k.title()}:[/bold] {v}"
                for k, v in char_details.items()
            ]),
            title="👤 Character Details",
            border_style="magenta",
            box=ROUNDED
        )
        left_panel_content.append(char_panel)

    # Colors panel with swatches
    if analysis.get('colors'):
        colors_panel = Panel(
            "\n".join([
                str(create_color_swatch(c['name'], c['hex_code'], c['prominence']))
                for c in sorted(analysis['colors'], key=lambda x: x['prominence'], reverse=True)
            ]),
            title="🎨 Color Palette",
            border_style="red",
            box=ROUNDED
        )
        left_panel_content.append(colors_panel)

    # Update left side
    left_content = Columns(left_panel_content, equal=True)
    main_layout["left"].update(left_content)

    # Right side - Objects and details
    right_panels = []
    
    if analysis.get('objects'):
        for obj in analysis['objects']:
            # Create confidence display with color coding
            confidence = obj['confidence']
            conf_color = "bright_green" if confidence > 0.8 else "yellow" if confidence > 0.6 else "red"
            confidence_text = f"[{conf_color}]{confidence:.1%}[/{conf_color}]"
            
            # Format object name with emoji based on type if possible
            obj_emoji = "🎯"  # Default
            obj_type = obj.get('type', '').lower()
            if "person" in obj_type or any(animal in obj_type for animal in ["human", "character", "boy", "girl", "man", "girl"]):
                obj_emoji = "👤"
            elif "animal" in obj_type or any(animal in obj_type for animal in ["dog", "cat", "bird"]):
                obj_emoji = "🐾"
            elif "vehicle" in obj_type or any(vehicle in obj_type for vehicle in ["car", "truck", "bike"]):
                obj_emoji = "🚗"
            elif "food" in obj_type:
                obj_emoji = "🍽️"
            elif "plant" in obj_type or "flower" in obj_type:
                obj_emoji = "🌿"
            
            # Create structured content with better spacing and organization
            obj_content = [
                f"[bold cyan]{obj['name']}[/bold cyan] {confidence_text}",
                ""  # Add space after title
            ]
            
            # Add description with improved formatting
            if obj.get('description'):
                obj_content.extend([
                    "[yellow underline]Description:[/yellow underline]",
                    f"[italic]{obj.get('description')}[/italic]",
                    ""  # Add space after section
                ])
            
            # Add location information with better formatting
            if obj.get('bounding_box'):
                box_info = obj.get('bounding_box')
                # Format bounding box as table-like structure if it's a dictionary
                if isinstance(box_info, dict):
                    box_text = "\n".join([f"  [dim]•[/dim] [cyan]{k}:[/cyan] {v}" for k, v in box_info.items()])
                else:
                    box_text = str(box_info)
                
                obj_content.extend([
                    "[yellow underline]Location:[/yellow underline]",
                    box_text,
                    ""  # Add space after section
                ])
            
            # Add attributes tree with improved styling
            if obj.get('attributes'):
                obj_content.append("[yellow underline]Attributes:[/yellow underline]")
                attr_text = create_attributes_tree(obj['attributes'])
                obj_content.append(attr_text)
            
            # Create panel with improved styling
            conf_style = "green" if confidence > 0.8 else "yellow" if confidence > 0.6 else "red"
            right_panels.append(
                Panel(
                    Padding(Text.from_markup("\n".join(obj_content)), (1, 2)),
                    title=f"{obj_emoji} {obj['name']}",
                    border_style=conf_style,
                    box=DOUBLE,
                    title_align="left",
                    subtitle=f"Confidence: {confidence:.1%}",
                    subtitle_align="right"
                )
            )

    # Arrange panels in a responsive grid layout
    if right_panels:
        # Use columns with proper spacing
        main_layout["right"].update(
            Panel(
                Columns(right_panels, equal=True, padding=(0, 1)),
                title="📋 Detected Objects",
                border_style="blue",
                box=ROUNDED
            )
        )
    else:
        main_layout["right"].update(
            Panel(
                "[italic]No objects detected in this image[/italic]",
                title="📋 Objects",
                border_style="blue",
                box=ROUNDED
            )
        )
        
    layout["main"].update(main_layout)

    # Footer with tags
    if analysis.get('tags'):
        tags_text = Text()
        for tag in analysis['tags']:
            tags_text.append(f"#{tag} ", style="bold blue")
        layout["footer"].update(Panel(tags_text, title="🏷️ Tags", border_style="blue"))

    # Display the layout
    console.print(layout)

    # Additional information for --full mode
    if full:
        console.print("\n[bold magenta]📝 Additional Details[/bold magenta]")
        
        # Technical details
        tech_table = Table(title="Technical Information", box=DOUBLE)
        tech_table.add_column("Property", style="cyan")
        tech_table.add_column("Value", style="green")
        
        tech_details = {
            "Image Path": analysis.get('image_path', 'Unknown'),
            "Suggested Filename": analysis.get('suggested_filename', 'None'),
            "Analysis Duration": f"{analysis.get('analysis_duration', 0):.2f}s",
            "Timestamp": analysis.get('timestamp', 'Unknown'),
            "Error Status": analysis.get('error', 'None')
        }
        
        for key, value in tech_details.items():
            tech_table.add_row(key, str(value))
        
        console.print(tech_table)


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze images using AI to detect objects, scenes, colors, and text."
    )

    # Command subparsers
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show analysis details for a specific image')
    show_parser.add_argument('-s', '--short', action='store_true', help='Show shortened output')
    show_parser.add_argument('-f', '--full', action='store_true', help='Show full detailed output')
    show_parser.add_argument('file', type=str, help='Image file to show analysis for')

    # Analyze command (existing functionality)
    analyze_parser = subparsers.add_parser('analyze', help='Analyze images')
    input_group = analyze_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-f", "--file", type=str, help="Single image file to analyze")
    input_group.add_argument("-d", "--directory", type=str, help="Directory containing images to analyze")
    
    # Rest of the existing arguments for analyze command
    analyze_parser.add_argument("-o", "--output", type=str, help="Output file path")
    analyze_parser.add_argument("--format", choices=["json", "txt"], default="json", help="Output format")
    analyze_parser.add_argument("-b", "--batch-size", type=int, default=10, help="Batch size for processing")
    analyze_parser.add_argument("--no-progress", action="store_true", help="Disable progress bar")

    args = parser.parse_args()

    if args.command == 'show':
        await display_image_analysis(args.file, full=args.full, short=args.short)
        return

    # Existing analyze logic
    if args.command == 'analyze':
        try:
            if args.file:
                image_path = Path(args.file)
                if not image_path.exists():
                    console.print(f"[red]Error: File not found: {image_path}[/red]")
                    return
                results = await process_single_file(image_path, args)
            else:
                image_dir = Path(args.directory)
                if not image_dir.exists():
                    console.print(f"[red]Error: Directory not found: {image_dir}[/red]")
                    return

                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                    TimeRemainingColumn(),
                    disable=args.no_progress,
                ) as progress:
                    # Load previous progress
                    progress_data = await load_progress()
                    results = progress_data["results"]
                    processed_files = progress_data["processed_files"]

                    # Get all image files
                    all_image_files = [
                        f
                        for f in image_dir.glob("*")
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif", ".bmp"}
                    ]

                    # Count total files including already processed ones
                    total_all_files = len(all_image_files)

                    # Filter out already processed files
                    remaining_files = [f for f in all_image_files if str(f) not in processed_files]
                    remaining_count = len(remaining_files)

                    if remaining_count == 0:
                        console.print(
                            f"[yellow]All {total_all_files} files have been processed already.[/yellow]"
                        )
                        return results

                    task = progress.add_task(
                        f"[cyan]Processing {remaining_count}/{total_all_files} remaining images...",
                        total=remaining_count,
                        completed=0
                    )

                    async def process_image(img_path):
                        try:
                            image_desc, analysis_meta = await analyze_image(img_path)
                            if image_desc:
                                results[img_path.name] = {
                                    **image_desc.model_dump(),
                                    **analysis_meta.model_dump(),
                                }
                            processed_files.append(str(img_path))

                            progress.update(
                                task,
                                advance=1,
                                description=f"[cyan]Processing {len(processed_files)}/{total_all_files} images..."
                            )

                            # Save progress after each image
                            await save_progress(results, processed_files)
                            return True
                        except Exception as e:
                            logger.error(f"Error processing image {img_path}: {str(e)}")
                            # Save progress even on failure
                            await save_progress(results, processed_files)
                            return False

                    # Create a semaphore to limit concurrent processing
                    sem = asyncio.Semaphore(args.batch_size)

                    async def process_with_semaphore(img_path):
                        async with sem:
                            return await process_image(img_path)

                    # Process all images concurrently with semaphore control
                    tasks = [process_with_semaphore(f) for f in remaining_files]

                    # Process in chunks to ensure regular saves
                    chunk_size = 10 # Save progress every 50 images
                    for i in range(0, len(tasks), chunk_size):
                        chunk = tasks[i:i + chunk_size]
                        await asyncio.gather(*chunk)
                        await save_progress(results, processed_files)  # Save after each chunk

                    # Save final results
                    await save_progress(results, processed_files)
                    if args.output:
                        save_results(results, args.output, args.format)

            # Print summary
            console.print(f"\n[green]Analysis complete![/green]")
            console.print(f"Total images processed: {len(results)}")
            success_count = sum(1 for r in results.values() if not r.get('error') and len(r) > 1)
            failed_count = sum(1 for r in results.values() if r.get('error') or len(r) <= 1)
            console.print(f"Successful: {success_count}")
            console.print(f"Failed: {failed_count}")
            console.print(f"Results saved to: {args.output}")

        except Exception as e:
            logger.critical("Fatal error in main process", exc_info=True)
            console.print(f"[red]Error: {str(e)}[/red]")
            raise
    else:
        console.print("[red]Invalid command. Use --help for more information.[/red]")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Fatal error: {str(e)}[/red]")