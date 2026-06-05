#!/usr/bin/env python3
"""Photo Bot Web Server - Flask backend for image/video generation."""

import os
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from pathlib import Path
from dotenv import load_dotenv

import database as db
import generator

load_dotenv()

app = Flask(__name__, static_folder='static')
CORS(app)

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


# ============== API Routes ==============

@app.route('/api/characters', methods=['GET'])
def list_characters():
    """List all characters."""
    characters = db.list_characters()
    return jsonify(characters)


@app.route('/api/characters', methods=['POST'])
def create_character():
    """Create a new character."""
    data = request.json
    name = data.get('name')
    description = data.get('description')
    traits = data.get('traits', [])

    if not name or not description:
        return jsonify({'error': 'Name and description are required'}), 400

    try:
        char_id = db.create_character(name, description, traits)
        return jsonify({'id': char_id, 'name': name, 'message': 'Character created'})
    except Exception as e:
        if "UNIQUE constraint" in str(e):
            return jsonify({'error': f"Character '{name}' already exists"}), 400
        return jsonify({'error': str(e)}), 500


@app.route('/api/characters/<name>', methods=['GET'])
def get_character(name):
    """Get a character by name."""
    char = db.get_character(name)
    if not char:
        return jsonify({'error': 'Character not found'}), 404
    return jsonify(char)


@app.route('/api/characters/<name>', methods=['PUT'])
def update_character(name):
    """Update a character."""
    data = request.json
    description = data.get('description')
    traits = data.get('traits')

    if db.update_character(name, description, traits):
        return jsonify({'message': 'Character updated'})
    return jsonify({'error': 'Character not found'}), 404


@app.route('/api/characters/<name>', methods=['DELETE'])
def delete_character(name):
    """Delete a character."""
    if db.delete_character(name):
        return jsonify({'message': 'Character deleted'})
    return jsonify({'error': 'Character not found'}), 404


@app.route('/api/characters/<name>/generations', methods=['GET'])
def get_character_generations(name):
    """Get all generations for a character."""
    char = db.get_character(name)
    if not char:
        return jsonify({'error': 'Character not found'}), 404

    generations = db.get_character_generations(char['id'])
    return jsonify(generations)


@app.route('/api/generate/image', methods=['POST'])
def generate_image():
    """Generate an image."""
    data = request.json
    prompt = data.get('prompt')
    character_name = data.get('character')
    model = data.get('model', 'flux-schnell')
    width = data.get('width', 1024)
    height = data.get('height', 1024)
    count = data.get('count', 1)
    negative = data.get('negative', '')
    guidance = data.get('guidance', 7.5)
    steps = data.get('steps', 25)

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    char = None
    char_id = None
    if character_name:
        char = db.get_character(character_name)
        if not char:
            return jsonify({'error': f"Character '{character_name}' not found"}), 404
        char_id = char['id']

    try:
        paths = generator.generate_image(
            prompt=prompt,
            character=char,
            model=model,
            width=width,
            height=height,
            num_outputs=count,
            negative_prompt=negative,
            guidance_scale=guidance,
            num_inference_steps=steps,
        )

        # Log to database and convert to URLs
        urls = []
        for path in paths:
            full_prompt = generator.build_prompt(prompt, char)
            db.log_generation(char_id, full_prompt, path, "image", model)
            # Return relative URL
            filename = Path(path).name
            urls.append(f'/outputs/{filename}')

        return jsonify({'images': urls, 'paths': paths})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/video', methods=['POST'])
def generate_video():
    """Generate a video."""
    data = request.json
    prompt = data.get('prompt', '')
    image_path = data.get('image_path')
    character_name = data.get('character')
    model = data.get('model', 'svd')
    frames = data.get('frames', 25)
    fps = data.get('fps', 6)

    char = None
    char_id = None
    if character_name:
        char = db.get_character(character_name)
        if not char:
            return jsonify({'error': f"Character '{character_name}' not found"}), 404
        char_id = char['id']

    try:
        path = generator.generate_video(
            prompt=prompt,
            image_path=image_path,
            character=char,
            model=model,
            frames=frames,
            fps=fps,
        )

        full_prompt = generator.build_prompt(prompt, char) if prompt else "(from image)"
        db.log_generation(char_id, full_prompt, path, "video", model)

        filename = Path(path).name
        return jsonify({'video': f'/outputs/{filename}', 'path': path})

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models."""
    return jsonify(generator.list_models())


# ============== Static File Serving ==============

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """Serve generated images/videos."""
    return send_from_directory(OUTPUTS_DIR, filename)


@app.route('/')
def serve_index():
    """Serve the main page."""
    return send_file('static/index.html')


@app.route('/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


if __name__ == '__main__':
    print("Starting Photo Bot server at http://localhost:5000")
    app.run(debug=True, port=5000)
