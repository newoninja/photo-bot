"""Image and video generation using multiple APIs (Replicate, xAI Grok, Google Gemini)."""

import os
import replicate
import requests
import base64
from pathlib import Path
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OUTPUTS_DIR = Path(__file__).parent / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# xAI Grok client
XAI_API_KEY = os.getenv("XAI_API_KEY")
grok_client = None
if XAI_API_KEY:
    grok_client = OpenAI(base_url="https://api.x.ai/v1", api_key=XAI_API_KEY)

# Google Gemini client for NanoBanana
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = None
if GOOGLE_API_KEY:
    from google import genai
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

# Image models - ordered by quality for realistic people
IMAGE_MODELS = {
    # NANOBANANA - Google Gemini image model
    "nanobanana": "gemini-2.5-flash-image",
    # GROK - xAI Aurora model (latest)
    "grok": "grok-imagine-image",
    # FLUX PRO models - photorealistic quality (RECOMMENDED)
    "flux-pro-ultra": "black-forest-labs/flux-1.1-pro-ultra:5ea10f739af9f6d4002fae9aee4c15be14c3c8d7f8b309e634bf68df09159863",
    "flux-pro": "black-forest-labs/flux-1.1-pro:609793a667ed94b210242837d3c3c9fc9a64ae93685f15d75002ba0ed9a97f2b",
    # FLUX open models - good quality, cheaper
    "flux-dev": "black-forest-labs/flux-dev:6e4a938f85952bdabcc15aa329178c4d681c52bf25a0342403287dc26944661d",
    "flux-schnell": "black-forest-labs/flux-schnell:c846a69991daf4c0e5d016514849d14ee5b2e6846ce6b9d6f21369e564cfe51e",
    # SDXL models - stylized/painted look
    "sdxl": "stability-ai/sdxl:7762fd07cf82c948538e41f63f77d685e02b063e37e496e96eefd46c929f9bdc",
    "juggernaut-xl": "lucataco/juggernaut-xl-v9:bea09cf018e513cef0841719559ea86d2299e05448633ac8fe270b5d5cd6777e",
    "realvisxl": "adirik/realvisxl-v4.0:85a58cc71587cc27539b7c83eb1ce4aea02feedfb9a9fae0598cebc110a3d695",
}

# Models that support multiple outputs natively
MULTI_OUTPUT_MODELS = {"flux-dev", "flux-schnell", "sdxl"}

# Default negative prompt for realistic models
DEFAULT_NEGATIVE = "deformed, distorted, disfigured, poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, mutated hands and fingers, disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, duplicate faces, multiple people, split image, collage"

VIDEO_MODELS = {
    "svd": "stability-ai/stable-video-diffusion:3f0457e4619daac51203dedb472816fd4af51f3149fa7a9e0b5ffcf1b8172438",
    "animate": "lucataco/animate-diff:beecf59c4aee8d81bf04f0381033dfa10dc16e845b4ae00d281e2fa377e48a9f",
}

DEFAULT_IMAGE_MODEL = "flux-pro-ultra"
DEFAULT_VIDEO_MODEL = "svd"


def build_prompt(base_prompt: str, character: Optional[dict] = None) -> str:
    """Build a full prompt incorporating character details."""
    if not character:
        return base_prompt

    parts = []

    # Add character description
    if character.get("description"):
        parts.append(character["description"])

    # Add traits
    if character.get("traits"):
        traits_str = ", ".join(character["traits"])
        parts.append(traits_str)

    # Add the user's prompt
    parts.append(base_prompt)

    return ", ".join(parts)


def generate_nanobanana_image(prompt: str, num_outputs: int = 1) -> list[str]:
    """Generate images using Google Gemini (NanoBanana) model."""
    if not gemini_client:
        raise ValueError("GOOGLE_API_KEY not set")

    from google.genai import types

    saved_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_outputs):
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=['IMAGE'],
            ),
        )

        # Get image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image_data = part.inline_data.data

                    filename = f"image_{timestamp}_{i}.png"
                    filepath = OUTPUTS_DIR / filename

                    with open(filepath, "wb") as f:
                        f.write(image_data)

                    saved_paths.append(str(filepath))
                    break

    return saved_paths


def generate_grok_image(prompt: str, num_outputs: int = 1) -> list[str]:
    """Generate images using xAI Grok Aurora model."""
    if not grok_client:
        raise ValueError("XAI_API_KEY not set")

    saved_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i in range(num_outputs):
        response = grok_client.images.generate(
            model="grok-imagine-image",
            prompt=prompt,
            n=1,
        )

        image_url = response.data[0].url

        # Download and save
        img_response = requests.get(image_url)
        img_response.raise_for_status()

        filename = f"image_{timestamp}_{i}.png"
        filepath = OUTPUTS_DIR / filename

        with open(filepath, "wb") as f:
            f.write(img_response.content)

        saved_paths.append(str(filepath))

    return saved_paths


def generate_image(
    prompt: str,
    character: Optional[dict] = None,
    model: str = DEFAULT_IMAGE_MODEL,
    width: int = 1024,
    height: int = 1024,
    num_outputs: int = 1,
    negative_prompt: str = "",
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
) -> list[str]:
    """
    Generate images using Replicate.

    Returns list of local file paths to saved images.
    """
    full_prompt = build_prompt(prompt, character)
    model_id = IMAGE_MODELS.get(model, model)

    # Use default negative prompt if none provided (for non-flux models)
    if not negative_prompt and "flux" not in model.lower() and "grok" not in model.lower():
        negative_prompt = DEFAULT_NEGATIVE

    # Handle NanoBanana separately (uses Google Gemini API)
    if model == "nanobanana":
        if not gemini_client:
            raise ValueError("GOOGLE_API_KEY not set. Add it to your .env file.")
        return generate_nanobanana_image(full_prompt, num_outputs)

    # Handle Grok separately (uses OpenAI-compatible API)
    if model == "grok":
        if not grok_client:
            raise ValueError("XAI_API_KEY not set. Add it to your .env file.")
        return generate_grok_image(full_prompt, num_outputs)

    # Determine aspect ratio
    if width == height:
        aspect = "1:1"
    elif width > height:
        aspect = "16:9" if width / height > 1.5 else "4:3"
    else:
        aspect = "9:16" if height / width > 1.5 else "3:4"

    # Build input based on model
    if "flux-pro" in model.lower():
        # Flux Pro / Pro Ultra - highest quality, photorealistic
        input_params = {
            "prompt": full_prompt,
            "aspect_ratio": aspect,
            "output_format": "png",
            "safety_tolerance": 6,  # Max permissiveness for NSFW
        }
        if "ultra" in model.lower():
            input_params["raw"] = True  # Raw mode for more photorealistic output
    elif "flux" in model.lower():
        # Flux Dev / Schnell
        input_params = {
            "prompt": full_prompt,
            "num_outputs": num_outputs,
            "aspect_ratio": aspect,
            "output_format": "png",
            "output_quality": 90,
        }
        if guidance_scale != 7.5:
            input_params["guidance"] = guidance_scale
    else:
        input_params = {
            "prompt": full_prompt,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "negative_prompt": negative_prompt,
            "disable_safety_checker": True,
        }
        # Only add num_outputs for models that support it
        if model in MULTI_OUTPUT_MODELS:
            input_params["num_outputs"] = num_outputs

    # For models that don't support multi-output, run multiple times
    if model not in MULTI_OUTPUT_MODELS and num_outputs > 1:
        all_outputs = []
        for _ in range(num_outputs):
            output = replicate.run(model_id, input=input_params)
            all_outputs.append(output)
        # Flatten outputs
        output = all_outputs
    else:
        output = replicate.run(model_id, input=input_params)

    # Save outputs locally
    saved_paths = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def download_and_save(url: str, index: int) -> str:
        """Download image from URL and save locally."""
        filename = f"image_{timestamp}_{index}.png"
        filepath = OUTPUTS_DIR / filename
        response = requests.get(url)
        response.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(response.content)
        return str(filepath)

    def process_single_output(item, index: int):
        """Process a single output item and save it."""
        if isinstance(item, str) and item.startswith('http'):
            return download_and_save(item, index)
        elif hasattr(item, 'url'):
            return download_and_save(item.url, index)
        else:
            url_str = str(item)
            if url_str.startswith('http'):
                return download_and_save(url_str, index)
        return None

    # Handle different output types from replicate
    idx = 0
    if isinstance(output, list):
        # List of outputs (from multiple runs or native multi-output)
        for item in output:
            # Each item could be a single output or an iterator
            if hasattr(item, '__iter__') and not isinstance(item, str) and not hasattr(item, 'url'):
                for sub_item in item:
                    path = process_single_output(sub_item, idx)
                    if path:
                        saved_paths.append(path)
                        idx += 1
            else:
                path = process_single_output(item, idx)
                if path:
                    saved_paths.append(path)
                    idx += 1
    elif isinstance(output, str):
        saved_paths.append(download_and_save(output, 0))
    elif hasattr(output, 'url'):
        saved_paths.append(download_and_save(output.url, 0))
    elif hasattr(output, '__iter__'):
        for item in output:
            path = process_single_output(item, idx)
            if path:
                saved_paths.append(path)
                idx += 1
    else:
        url_str = str(output)
        if url_str.startswith('http'):
            saved_paths.append(download_and_save(url_str, 0))

    return saved_paths


def generate_video(
    prompt: str = "",
    image_path: Optional[str] = None,
    character: Optional[dict] = None,
    model: str = DEFAULT_VIDEO_MODEL,
    frames: int = 25,
    fps: int = 6,
) -> str:
    """
    Generate video using Replicate.

    For SVD model, requires an input image.
    For animate-diff, uses text prompt.

    Returns local file path to saved video.
    """
    model_id = VIDEO_MODELS.get(model, model)
    full_prompt = build_prompt(prompt, character) if prompt else ""

    if model == "svd":
        if not image_path:
            raise ValueError("SVD model requires an input image. Generate an image first.")

        # Upload image to get URL
        with open(image_path, "rb") as f:
            input_params = {
                "input_image": f,
                "frames_per_second": fps,
                "motion_bucket_id": 127,  # Controls motion amount
            }
            output = replicate.run(model_id, input=input_params)

    elif model == "animate":
        input_params = {
            "prompt": full_prompt,
            "num_frames": frames,
            "fps": fps,
        }
        output = replicate.run(model_id, input=input_params)
    else:
        # Generic model handling
        input_params = {"prompt": full_prompt}
        if image_path:
            with open(image_path, "rb") as f:
                input_params["image"] = f
        output = replicate.run(model_id, input=input_params)

    # Handle output - could be URL or FileOutput
    if hasattr(output, 'url'):
        video_url = output.url
    elif hasattr(output, 'read'):
        content = output.read()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}.mp4"
        filepath = OUTPUTS_DIR / filename
        with open(filepath, "wb") as f:
            f.write(content)
        return str(filepath)
    elif isinstance(output, str):
        video_url = output
    else:
        # Iterator or list
        video_url = list(output)[0]
        if hasattr(video_url, 'url'):
            video_url = video_url.url

    # Download video
    response = requests.get(video_url)
    response.raise_for_status()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"video_{timestamp}.mp4"
    filepath = OUTPUTS_DIR / filename

    with open(filepath, "wb") as f:
        f.write(response.content)

    return str(filepath)


def list_models():
    """Return available models."""
    return {
        "image": list(IMAGE_MODELS.keys()),
        "video": list(VIDEO_MODELS.keys()),
    }
