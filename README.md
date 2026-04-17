# MemeScope

**Multimodal Cultural Context Explanation Using Vision Language Models**

CSE 434/534: Generative AI | Miami University
Team: Himank Juttiga, August Friedrich, Andrew LaPlante

## Overview

MemeScope is a multimodal pipeline that takes an internet meme image as input and generates a structured plain-English explanation covering the visual content, embedded text meaning, and cultural context. The system combines OCR, vision-language captioning, and large language model prompting to bridge the gap between visual internet culture and accessible language.

## Architecture

```
Meme Image
    |
    +---> EasyOCR ---------> Extracted Text
    |                              |
    +---> BLIP-2 / Claude -----> Visual Caption
                                   |
                          Structured Prompt
                                   |
                     GPT-3.5 / LLaMA-2 / Claude
                                   |
                      Cultural Context Explanation
```

## Project Structure

```
memescope/
├── pipeline/
│   ├── memescope.py       # Core pipeline (OCR + Vision + LLM)
│   └── finetune.py        # LoRA/QLoRA fine-tuning for LLaMA-2 (CSE 534)
├── evaluation/
│   └── evaluate.py        # ROUGE-L and BERTScore evaluation
├── demo/
│   └── app.py             # Gradio web demo
├── data/                  # Place MemeCap dataset here
├── outputs/               # Pipeline results and checkpoints
├── docs/
│   ├── Project_Proposal_MemeScope.pdf
│   └── MemeScope_Midterm_Report.pdf
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/memescope.git
cd memescope

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY="your_key_here"
```

## Usage

### Explain a single meme

```bash
python pipeline/memescope.py path/to/meme.jpg
```

Options:
  * `--strategy zero_shot` or `--strategy few_shot` (default: few_shot)
  * `--api-key YOUR_KEY` (or set ANTHROPIC_API_KEY env var)

### Run the Gradio demo

```bash
python demo/app.py
```

Then open `http://localhost:7860` in your browser.

### Evaluate against references

```bash
python evaluation/evaluate.py outputs/results.json data/references.json
```

### Fine-tune LLaMA-2 with LoRA (CSE 534)

```bash
export HF_TOKEN="your_huggingface_token"
python pipeline/finetune.py
```

Requires a CUDA-capable GPU with at least 16GB VRAM.

## Datasets

| Dataset | Purpose | Source |
|---------|---------|--------|
| MemeCap | Primary training and evaluation | [MemeCap GitHub](https://github.com/eujhwang/meme-cap) |
| Reddit Pushshift | Out-of-distribution testing | [Pushshift.io](https://pushshift.io) |

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| ROUGE-L | Longest common subsequence overlap with references |
| BERTScore | Semantic similarity using contextual embeddings |
| Human Rating | 1 to 5 scale across visual accuracy, text accuracy, and humor explanation |

## Models Used

| Model | Role |
|-------|------|
| EasyOCR | Embedded text extraction from meme images |
| BLIP-2 / Claude Vision | Visual captioning of image content |
| GPT-3.5 / LLaMA-2 / Claude | Cultural context explanation generation |
| LLaMA-2 + LoRA | Fine-tuned variant for improved explanation quality |

## License

This project is for educational purposes as part of CSE 434/534 at Miami University.
