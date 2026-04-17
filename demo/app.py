"""
MemeScope Gradio Demo
Upload a meme image and get a structured cultural explanation.
"""

import os
import sys
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.memescope import explain_meme


API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")


def run_explanation(image_path, strategy):
    """Process a meme image and return the explanation."""
    if not API_KEY:
        return (
            "API key not set.",
            "Please set the ANTHROPIC_API_KEY environment variable.",
            "",
            "",
        )
    if image_path is None:
        return ("No image uploaded.", "", "", "")

    try:
        result = explain_meme(image_path, API_KEY, strategy)
        return (
            result.get("ocr_text", ""),
            result.get("visual", ""),
            result.get("text_meaning", ""),
            result.get("cultural_context", ""),
        )
    except Exception as e:
        return (f"Error: {str(e)}", "", "", "")


with gr.Blocks(
    title="MemeScope",
    theme=gr.themes.Soft(primary_hue="red"),
) as demo:
    gr.Markdown("# MemeScope")
    gr.Markdown(
        "Multimodal cultural context explanation for internet memes. "
        "Upload a meme and select a prompting strategy to get started."
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Meme Image")
            strategy_input = gr.Radio(
                choices=["zero_shot", "few_shot"],
                value="few_shot",
                label="Prompting Strategy",
            )
            run_btn = gr.Button("Explain This Meme", variant="primary")

        with gr.Column(scale=1):
            ocr_output = gr.Textbox(label="OCR Text Detected", lines=2)
            visual_output = gr.Textbox(label="Visual Description", lines=3)
            text_output = gr.Textbox(label="Text Meaning", lines=3)
            cultural_output = gr.Textbox(label="Cultural Context", lines=4)

    run_btn.click(
        fn=run_explanation,
        inputs=[image_input, strategy_input],
        outputs=[ocr_output, visual_output, text_output, cultural_output],
    )

    gr.Markdown(
        "---\n"
        "CSE 434/534: Generative AI | Miami University | "
        "Himank Juttiga, August Friedrich, Andrew LaPlante"
    )


if __name__ == "__main__":
    demo.launch(share=False)
