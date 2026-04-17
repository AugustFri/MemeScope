"""
MemeScope LoRA Fine-Tuning Script (CSE 534 Graduate Component)
Fine-tunes LLaMA-2-7B on MemeCap dataset using QLoRA for efficient training.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    model_name: str = "meta-llama/Llama-2-7b-hf"
    dataset_path: str = "data/memecap_train.json"
    output_dir: str = "outputs/lora_checkpoints"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "v_proj", "k_proj", "o_proj")
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    load_in_4bit: bool = True          # QLoRA
    bnb_4bit_compute_dtype: str = "float16"
    warmup_ratio: float = 0.03
    lr_scheduler: str = "cosine"
    save_steps: int = 100
    logging_steps: int = 25
    hf_token: str = ""                 # Set via HF_TOKEN env var


INSTRUCTION_TEMPLATE = """Below is a meme image description and any embedded text detected via OCR.
Write a structured explanation covering the visual content, the text meaning, and the cultural context.

### OCR Text:
{ocr_text}

### Visual Caption:
{visual_caption}

### Explanation:
VISUAL: {visual}
TEXT: {text_meaning}
CULTURAL CONTEXT: {cultural_context}"""


def format_dataset_entry(entry: dict) -> str:
    """Convert a MemeCap entry into the instruction-tuning format."""
    return INSTRUCTION_TEMPLATE.format(
        ocr_text=entry.get("ocr_text", "[None]"),
        visual_caption=entry.get("visual_caption", "[None]"),
        visual=entry.get("visual", ""),
        text_meaning=entry.get("text_meaning", ""),
        cultural_context=entry.get("cultural_context", ""),
    )


def prepare_dataset(dataset_path: str) -> list:
    """Load and format the MemeCap dataset for fine-tuning."""
    import json
    with open(dataset_path) as f:
        data = json.load(f)
    return [{"text": format_dataset_entry(entry)} for entry in data]


def load_model_and_tokenizer(config: TrainingConfig):
    """Load the base model with 4-bit quantization for QLoRA."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_auth_token=config.hf_token,
        padding_side="right",
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        use_auth_token=config.hf_token,
    )
    return model, tokenizer


def apply_lora(model, config: TrainingConfig):
    """Attach LoRA adapters to the model."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def train(config: TrainingConfig = None):
    """Run the full QLoRA fine-tuning job."""
    import os
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from datasets import Dataset

    if config is None:
        config = TrainingConfig()

    config.hf_token = os.environ.get("HF_TOKEN", config.hf_token)

    print("[MemeScope LoRA] Loading model...")
    model, tokenizer = load_model_and_tokenizer(config)
    model = apply_lora(model, config)
    model.print_trainable_parameters()

    print("[MemeScope LoRA] Preparing dataset...")
    formatted = prepare_dataset(config.dataset_path)
    dataset = Dataset.from_list(formatted)

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler,
        warmup_ratio=config.warmup_ratio,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )

    print("[MemeScope LoRA] Starting training...")
    trainer.train()

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"[MemeScope LoRA] Saved to {config.output_dir}")


if __name__ == "__main__":
    train()
