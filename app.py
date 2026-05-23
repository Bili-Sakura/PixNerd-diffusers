import argparse
import sys
from pathlib import Path

import gradio as gr
import torch

REPO_SRC = Path(__file__).resolve().parent / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from diffusers import PixNerdPipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="models/BiliSakura/PixNerd-diffusers/PixNerd-XL-16-256",
    )
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    pipeline = PixNerdPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    pipeline.scheduler.timeshift = float(args.timeshift)
    pipeline.scheduler.order = int(args.order)

    def generate(class_label, num_images, seed, image_height, image_width, num_steps, guidance, timeshift, order):
        label = int(class_label) if str(class_label).strip().isdigit() else class_label
        pipeline.scheduler.timeshift = float(timeshift)
        pipeline.scheduler.order = int(order)
        images = pipeline(
            class_labels=label,
            num_images_per_prompt=int(num_images),
            generator=torch.Generator(device=args.device).manual_seed(int(seed)),
            height=int(image_height),
            width=int(image_width),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance),
            output_type="pil",
        ).images
        return images

    with gr.Blocks() as demo:
        gr.Markdown(f"pretrained_model_name_or_path: {args.pretrained_model_name_or_path}")
        with gr.Row():
            with gr.Column(scale=1):
                num_steps = gr.Slider(minimum=1, maximum=100, step=1, label="num steps", value=25)
                guidance = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, label="CFG", value=4.0)
                image_height = gr.Slider(minimum=128, maximum=1024, step=16, label="image height", value=256)
                image_width = gr.Slider(minimum=128, maximum=1024, step=16, label="image width", value=256)
                num_images = gr.Slider(minimum=1, maximum=4, step=1, label="num images", value=1)
                label = gr.Textbox(label="class label (string or integer id)", value="golden retriever")
                seed = gr.Slider(minimum=0, maximum=1000000, step=1, label="seed", value=42)
                timeshift = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, label="timeshift", value=3.0)
                order = gr.Slider(minimum=1, maximum=4, step=1, label="order", value=2)
            with gr.Column(scale=2):
                btn = gr.Button("Generate")
                output_sample = gr.Gallery(label="Images", columns=2, rows=2)

        btn.click(
            fn=generate,
            inputs=[
                label,
                num_images,
                seed,
                image_height,
                image_width,
                num_steps,
                guidance,
                timeshift,
                order,
            ],
            outputs=[output_sample],
        )
    demo.launch(server_name="0.0.0.0", server_port=7861)
