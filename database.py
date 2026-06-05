"""SQLite database for storing characters."""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

DB_PATH = Path(__file__).parent / "characters.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the database with required tables."""
    conn = get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS characters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            traits TEXT,
            reference_images TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            character_id INTEGER,
            prompt TEXT NOT NULL,
            output_path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            model_used TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (character_id) REFERENCES characters(id)
        )
    """)
    conn.commit()
    conn.close()


def create_character(
    name: str,
    description: str,
    traits: Optional[list[str]] = None,
    reference_images: Optional[list[str]] = None
) -> int:
    """Create a new character and return its ID."""
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO characters (name, description, traits, reference_images)
        VALUES (?, ?, ?, ?)
        """,
        (
            name,
            description,
            json.dumps(traits or []),
            json.dumps(reference_images or [])
        )
    )
    conn.commit()
    char_id = cursor.lastrowid
    conn.close()
    return char_id


def get_character(name: str) -> Optional[dict]:
    """Get a character by name."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM characters WHERE name = ?", (name,)
    ).fetchone()
    conn.close()

    if row:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "traits": json.loads(row["traits"]),
            "reference_images": json.loads(row["reference_images"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }
    return None


def get_character_by_id(char_id: int) -> Optional[dict]:
    """Get a character by ID."""
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM characters WHERE id = ?", (char_id,)
    ).fetchone()
    conn.close()

    if row:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "traits": json.loads(row["traits"]),
            "reference_images": json.loads(row["reference_images"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }
    return None


def list_characters() -> list[dict]:
    """List all characters."""
    conn = get_connection()
    rows = conn.execute("SELECT * FROM characters ORDER BY name").fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "traits": json.loads(row["traits"]),
            "reference_images": json.loads(row["reference_images"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"]
        }
        for row in rows
    ]


def update_character(
    name: str,
    description: Optional[str] = None,
    traits: Optional[list[str]] = None,
    reference_images: Optional[list[str]] = None
) -> bool:
    """Update an existing character."""
    char = get_character(name)
    if not char:
        return False

    conn = get_connection()
    conn.execute(
        """
        UPDATE characters
        SET description = ?,
            traits = ?,
            reference_images = ?,
            updated_at = ?
        WHERE name = ?
        """,
        (
            description if description is not None else char["description"],
            json.dumps(traits) if traits is not None else json.dumps(char["traits"]),
            json.dumps(reference_images) if reference_images is not None else json.dumps(char["reference_images"]),
            datetime.now().isoformat(),
            name
        )
    )
    conn.commit()
    conn.close()
    return True


def delete_character(name: str) -> bool:
    """Delete a character by name."""
    conn = get_connection()
    cursor = conn.execute("DELETE FROM characters WHERE name = ?", (name,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted


def log_generation(
    character_id: Optional[int],
    prompt: str,
    output_path: str,
    media_type: str,
    model_used: str
) -> int:
    """Log a generation to the database."""
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO generations (character_id, prompt, output_path, media_type, model_used)
        VALUES (?, ?, ?, ?, ?)
        """,
        (character_id, prompt, output_path, media_type, model_used)
    )
    conn.commit()
    gen_id = cursor.lastrowid
    conn.close()
    return gen_id


def get_character_generations(character_id: int) -> list[dict]:
    """Get all generations for a character."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM generations WHERE character_id = ? ORDER BY created_at DESC",
        (character_id,)
    ).fetchall()
    conn.close()

    return [
        {
            "id": row["id"],
            "prompt": row["prompt"],
            "output_path": row["output_path"],
            "media_type": row["media_type"],
            "model_used": row["model_used"],
            "created_at": row["created_at"]
        }
        for row in rows
    ]


# Initialize DB on import
init_db()
