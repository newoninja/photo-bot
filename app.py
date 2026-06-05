#!/usr/bin/env python3
"""Photo Bot CLI - Generate images and videos with persistent characters."""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

import database as db
import generator

console = Console()


@click.group()
def cli():
    """Photo Bot - Generate images and videos with your custom characters."""
    pass


# ============== Character Commands ==============

@cli.group()
def char():
    """Manage characters."""
    pass


@char.command("create")
@click.argument("name")
@click.option("--description", "-d", required=True, help="Physical description of the character")
@click.option("--trait", "-t", multiple=True, help="Character traits (can specify multiple)")
def char_create(name: str, description: str, trait: tuple):
    """Create a new character."""
    traits = list(trait) if trait else []

    try:
        char_id = db.create_character(name, description, traits)
        console.print(f"[green]Created character '{name}' with ID {char_id}[/green]")
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            console.print(f"[red]Character '{name}' already exists[/red]")
        else:
            console.print(f"[red]Error: {e}[/red]")


@char.command("list")
def char_list():
    """List all characters."""
    characters = db.list_characters()

    if not characters:
        console.print("[yellow]No characters found. Create one with: photobot char create[/yellow]")
        return

    table = Table(title="Characters")
    table.add_column("ID", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description", style="white", max_width=50)
    table.add_column("Traits", style="yellow")

    for char in characters:
        traits = ", ".join(char["traits"]) if char["traits"] else "-"
        table.add_row(
            str(char["id"]),
            char["name"],
            char["description"][:50] + "..." if len(char["description"]) > 50 else char["description"],
            traits
        )

    console.print(table)


@char.command("show")
@click.argument("name")
def char_show(name: str):
    """Show details of a character."""
    char = db.get_character(name)

    if not char:
        console.print(f"[red]Character '{name}' not found[/red]")
        return

    panel_content = f"""[bold]Name:[/bold] {char['name']}
[bold]ID:[/bold] {char['id']}
[bold]Description:[/bold] {char['description']}
[bold]Traits:[/bold] {', '.join(char['traits']) if char['traits'] else 'None'}
[bold]Created:[/bold] {char['created_at']}"""

    console.print(Panel(panel_content, title=f"Character: {char['name']}"))

    # Show recent generations
    generations = db.get_character_generations(char["id"])
    if generations:
        console.print(f"\n[bold]Recent generations ({len(generations)} total):[/bold]")
        for gen in generations[:5]:
            console.print(f"  - [{gen['media_type']}] {gen['output_path']}")


@char.command("update")
@click.argument("name")
@click.option("--description", "-d", help="New description")
@click.option("--trait", "-t", multiple=True, help="New traits (replaces existing)")
def char_update(name: str, description: str, trait: tuple):
    """Update a character."""
    traits = list(trait) if trait else None

    if db.update_character(name, description, traits):
        console.print(f"[green]Updated character '{name}'[/green]")
    else:
        console.print(f"[red]Character '{name}' not found[/red]")


@char.command("delete")
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to delete this character?")
def char_delete(name: str):
    """Delete a character."""
    if db.delete_character(name):
        console.print(f"[green]Deleted character '{name}'[/green]")
    else:
        console.print(f"[red]Character '{name}' not found[/red]")


# ============== Generation Commands ==============

@cli.command("image")
@click.argument("prompt")
@click.option("--character", "-c", help="Character name to use")
@click.option("--model", "-m", default="flux-schnell", help="Model to use (sdxl, realistic, flux-dev, flux-schnell)")
@click.option("--width", "-w", default=1024, help="Image width")
@click.option("--height", "-h", default=1024, help="Image height")
@click.option("--count", "-n", default=1, help="Number of images to generate")
@click.option("--negative", help="Negative prompt")
@click.option("--guidance", "-g", default=7.5, help="Guidance scale")
@click.option("--steps", "-s", default=25, help="Inference steps")
def generate_image(
    prompt: str,
    character: str,
    model: str,
    width: int,
    height: int,
    count: int,
    negative: str,
    guidance: float,
    steps: int
):
    """Generate an image."""
    char = None
    char_id = None

    if character:
        char = db.get_character(character)
        if not char:
            console.print(f"[red]Character '{character}' not found[/red]")
            return
        char_id = char["id"]
        console.print(f"[cyan]Using character: {character}[/cyan]")

    console.print(f"[yellow]Generating image with {model}...[/yellow]")

    try:
        paths = generator.generate_image(
            prompt=prompt,
            character=char,
            model=model,
            width=width,
            height=height,
            num_outputs=count,
            negative_prompt=negative or "",
            guidance_scale=guidance,
            num_inference_steps=steps,
        )

        for path in paths:
            console.print(f"[green]Saved: {path}[/green]")

            # Log to database
            full_prompt = generator.build_prompt(prompt, char)
            db.log_generation(char_id, full_prompt, path, "image", model)

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("video")
@click.option("--prompt", "-p", default="", help="Text prompt for video")
@click.option("--image", "-i", help="Input image path (required for SVD)")
@click.option("--character", "-c", help="Character name to use")
@click.option("--model", "-m", default="svd", help="Model to use (svd, animate)")
@click.option("--frames", "-f", default=25, help="Number of frames")
@click.option("--fps", default=6, help="Frames per second")
def generate_video(
    prompt: str,
    image: str,
    character: str,
    model: str,
    frames: int,
    fps: int
):
    """Generate a video."""
    char = None
    char_id = None

    if character:
        char = db.get_character(character)
        if not char:
            console.print(f"[red]Character '{character}' not found[/red]")
            return
        char_id = char["id"]
        console.print(f"[cyan]Using character: {character}[/cyan]")

    console.print(f"[yellow]Generating video with {model}...[/yellow]")

    try:
        path = generator.generate_video(
            prompt=prompt,
            image_path=image,
            character=char,
            model=model,
            frames=frames,
            fps=fps,
        )

        console.print(f"[green]Saved: {path}[/green]")

        # Log to database
        full_prompt = generator.build_prompt(prompt, char) if prompt else "(from image)"
        db.log_generation(char_id, full_prompt, path, "video", model)

    except ValueError as e:
        console.print(f"[red]{e}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("models")
def show_models():
    """Show available models."""
    models = generator.list_models()

    console.print("\n[bold]Image Models:[/bold]")
    for name in models["image"]:
        console.print(f"  - {name}")

    console.print("\n[bold]Video Models:[/bold]")
    for name in models["video"]:
        console.print(f"  - {name}")


# ============== Quick Generate (no character) ==============

@cli.command("quick")
@click.argument("prompt")
@click.option("--model", "-m", default="flux-schnell", help="Model to use")
def quick_generate(prompt: str, model: str):
    """Quick image generation without a character."""
    console.print(f"[yellow]Generating with {model}...[/yellow]")

    try:
        paths = generator.generate_image(prompt=prompt, model=model)
        for path in paths:
            console.print(f"[green]Saved: {path}[/green]")
            db.log_generation(None, prompt, path, "image", model)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    cli()
