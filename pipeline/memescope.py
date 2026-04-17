"""
MemeScope Pipeline
Multimodal Cultural Context Explanation Using Vision Language Models
Team: Himank Juttiga, August Friedrich, Andrew LaPlante
CSE 434/534 - Miami University
"""

import base64
import json
import re
import requests
from pathlib import Path
from PIL import Image
import io

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

# ── Prompt Templates ─────────────────────────────────────────────────────────

ZERO_SHOT_PROMPT = """You are a cultural context expert specializing in internet memes.

You are given:
- A meme image
- Any text extracted from the meme via OCR: "{ocr_text}"

Your task is to produce a structured plain-English explanation with exactly these three parts:

1. VISUAL: Describe what is shown in the image (template, characters, scene).
2. TEXT: Explain what the embedded text means in context.
3. CULTURAL CONTEXT: Explain why this is funny or culturally significant, including the meme template origin if known.

Be concise and clear. Assume the reader is unfamiliar with internet culture."""

FEW_SHOT_PROMPT = """You are a cultural context expert specializing in internet memes.

Here are two examples of good meme explanations:

EXAMPLE 1:
OCR Text: "Me | My diet | Free pizza at the office"
VISUAL: The 'Distracted Boyfriend' meme showing a man turning away from his girlfriend to look at another woman.
TEXT: The man represents the person, the girlfriend represents their diet, and the other woman represents free pizza at the office.
CULTURAL CONTEXT: This template originated from a 2015 stock photo by Antonio Guillem and became a viral meme in 2017. It is used to humorously illustrate someone being tempted by something they should not want, especially when they have an existing commitment.

EXAMPLE 2:
OCR Text: "This is fine"
VISUAL: A cartoon dog sitting calmly in a room that is entirely on fire, sipping coffee.
TEXT: The phrase 'This is fine' contrasts absurdly with the clearly catastrophic situation around the character.
CULTURAL CONTEXT: This comic panel by KC Green (2013) became a widely used meme to express resigned acceptance of a bad or chaotic situation, often used to comment on ongoing crises with dark humor.

Now explain this meme:
OCR Text: "{ocr_text}"

Provide your explanation in the same format:
VISUAL:
TEXT:
CULTURAL CONTEXT:"""


# ── Image Utilities ───────────────────────────────────────────────────────────

def load_image_as_base64(image_path: str) -> tuple[str, str]:
    """Load an image and return (base64_string, media_type)."""
    path = Path(image_path)
    ext = path.suffix.lower()
    media_type_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_type_map.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")
    return image_data, media_type


def extract_ocr_text(image_path: str) -> str:
    """Extract text from image using EasyOCR."""
    try:
        import easyocr
        reader = easyocr.Reader(["en"], verbose=False)
        results = reader.readtext(image_path)
        text = " ".join([res[1] for res in results])
        return text.strip() if text.strip() else "[No text detected]"
    except Exception as e:
        return f"[OCR unavailable: {e}]"


# ── Core API Call ─────────────────────────────────────────────────────────────

def call_claude(image_b64: str, media_type: str, prompt: str, api_key: str) -> str:
    """Send image + prompt to Claude and return the explanation text."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 600,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }
    response = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["content"][0]["text"]


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def explain_meme(
    image_path: str,
    api_key: str,
    strategy: str = "few_shot",
) -> dict:
    """
    Full MemeScope pipeline.

    Args:
        image_path: Path to the meme image file.
        api_key: Anthropic API key.
        strategy: 'zero_shot' or 'few_shot'

    Returns:
        dict with keys: ocr_text, explanation, visual, text_meaning, cultural_context
    """
    print(f"[MemeScope] Loading image: {image_path}")
    image_b64, media_type = load_image_as_base64(image_path)

    print("[MemeScope] Running OCR...")
    ocr_text = extract_ocr_text(image_path)
    print(f"[MemeScope] OCR result: {ocr_text}")

    if strategy == "zero_shot":
        prompt = ZERO_SHOT_PROMPT.format(ocr_text=ocr_text)
    else:
        prompt = FEW_SHOT_PROMPT.format(ocr_text=ocr_text)

    print(f"[MemeScope] Calling model with {strategy} strategy...")
    raw_output = call_claude(image_b64, media_type, prompt, api_key)

    # Parse structured output
    def extract_section(label, text):
        pattern = rf"{label}:\s*(.*?)(?=\n[A-Z ]+:|$)"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    result = {
        "ocr_text": ocr_text,
        "strategy": strategy,
        "raw_output": raw_output,
        "visual": extract_section("VISUAL", raw_output),
        "text_meaning": extract_section("TEXT", raw_output),
        "cultural_context": extract_section("CULTURAL CONTEXT", raw_output),
    }

    return result


def explain_meme_batch(
    image_paths: list,
    api_key: str,
    strategies: list = None,
    output_file: str = "outputs/results.json",
) -> list:
    """Run the pipeline on a batch of images and save results."""
    if strategies is None:
        strategies = ["zero_shot", "few_shot"]

    all_results = []
    for path in image_paths:
        entry = {"image": path, "results": {}}
        for strategy in strategies:
            try:
                result = explain_meme(path, api_key, strategy)
                entry["results"][strategy] = result
            except Exception as e:
                entry["results"][strategy] = {"error": str(e)}
        all_results.append(entry)
        print(f"[MemeScope] Completed: {path}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[MemeScope] Results saved to {output_file}")

    return all_results


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="MemeScope: Explain a meme image")
    parser.add_argument("image", help="Path to the meme image")
    parser.add_argument("--strategy", default="few_shot", choices=["zero_shot", "few_shot"])
    parser.add_argument("--api-key", default=os.environ.get("ANTHROPIC_API_KEY", ""))
    args = parser.parse_args()

    if not args.api_key:
        print("Error: Set ANTHROPIC_API_KEY environment variable or pass --api-key")
        exit(1)

    result = explain_meme(args.image, args.api_key, args.strategy)

    print("\n" + "="*60)
    print("MEMESCOPE EXPLANATION")
    print("="*60)
    print(f"OCR Text Detected: {result['ocr_text']}")
    print(f"\nVISUAL:\n{result['visual']}")
    print(f"\nTEXT MEANING:\n{result['text_meaning']}")
    print(f"\nCULTURAL CONTEXT:\n{result['cultural_context']}")
    print("="*60)
