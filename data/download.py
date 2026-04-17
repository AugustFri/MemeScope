"""
MemeScope Dataset Downloader
Downloads and prepares the MemeCap dataset for training and evaluation.
"""

import os
import json
import subprocess
from pathlib import Path


MEMECAP_REPO = "https://github.com/eujhwang/meme-cap.git"
DATA_DIR = Path(__file__).parent


def download_memecap():
    """Clone the MemeCap repository and extract the dataset."""
    clone_dir = DATA_DIR / "meme-cap"

    if clone_dir.exists():
        print("[Data] MemeCap already downloaded. Skipping clone.")
    else:
        print("[Data] Cloning MemeCap repository...")
        subprocess.run(["git", "clone", MEMECAP_REPO, str(clone_dir)], check=True)

    print("[Data] MemeCap ready at:", clone_dir)
    return clone_dir


def prepare_splits(memecap_dir: Path, output_dir: Path = None):
    """
    Prepare train/val/test splits from MemeCap data.
    Adjust this function based on the actual MemeCap data format.
    """
    if output_dir is None:
        output_dir = DATA_DIR

    print("[Data] Preparing splits...")
    print("[Data] Check the MemeCap repo for the exact JSON format,")
    print("       then adapt this function to produce:")
    print("       - memecap_train.json (5000 examples)")
    print("       - memecap_val.json   (750 examples)")
    print("       - memecap_test.json  (750 examples)")
    print()
    print("[Data] Each entry should have:")
    print("       {")
    print('         "image_path": "path/to/image.jpg",')
    print('         "ocr_text": "text from the meme",')
    print('         "visual_caption": "BLIP-2 generated caption",')
    print('         "visual": "human written visual description",')
    print('         "text_meaning": "human written text explanation",')
    print('         "cultural_context": "human written cultural context"')
    print("       }")


def prepare_references(test_file: str, output_file: str = None):
    """
    Create a references JSON for evaluation from the test split.
    """
    if output_file is None:
        output_file = str(DATA_DIR / "references.json")

    with open(test_file) as f:
        test_data = json.load(f)

    references = []
    for entry in test_data:
        ref_text = " ".join([
            entry.get("visual", ""),
            entry.get("text_meaning", ""),
            entry.get("cultural_context", ""),
        ]).strip()
        references.append({
            "image": entry["image_path"],
            "reference": ref_text,
        })

    with open(output_file, "w") as f:
        json.dump(references, f, indent=2)
    print(f"[Data] References saved to {output_file} ({len(references)} entries)")


if __name__ == "__main__":
    memecap = download_memecap()
    prepare_splits(memecap)
