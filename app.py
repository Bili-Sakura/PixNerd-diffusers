import os
import argparse
import torch
import gradio as gr
from huggingface_hub import snapshot_download

from src.pixnerd_diffusers.pipeline import PixNerdPipeline
from src.pixnerd_diffusers.scheduler import PixNerdFlowMatchScheduler
from src.pixnerd_diffusers.transformer import PixNerdTransformer2DModel
from src.pixnerd_diffusers.config_utils import to_container
from omegaconf import OmegaConf


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs_t2i/inference_heavydecoder.yaml")
    parser.add_argument("--model_id", type=str, default="MCG-NJU/PixNerd-XXL-P16-T2I")
    parser.add_argument("--ckpt_path", type=str, default="models/model.ckpt")
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    if not os.path.exists(args.ckpt_path):
        local_dir = os.path.dirname(args.ckpt_path) or "models"
        snapshot_download(repo_id=args.model_id, local_dir=local_dir)
        ckpt_path = os.path.join(local_dir, "model.ckpt")
    else:
        ckpt_path = args.ckpt_path

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    config = OmegaConf.load(args.config)
    model_cfg = to_container(config.model)
    transformer = PixNerdTransformer2DModel.from_project_config(model_cfg, use_ema=True)
    transformer.load_legacy_checkpoint(ckpt_path)
    transformer = transformer.to(dtype=dtype, device=args.device)
    sampler_cfg = model_cfg.get("diffusion_sampler", {}).get("init_args", {})
    scheduler = PixNerdFlowMatchScheduler(
        num_inference_steps=sampler_cfg.get("num_steps", 25),
        guidance_scale=sampler_cfg.get("guidance", 4.0),
        timeshift=sampler_cfg.get("timeshift", 3.0),
        order=sampler_cfg.get("order", 2),
        guidance_interval_min=sampler_cfg.get("guidance_interval_min", 0.0),
        guidance_interval_max=sampler_cfg.get("guidance_interval_max", 1.0),
        last_step=sampler_cfg.get("last_step", None),
    )
    pipeline = PixNerdPipeline(
        vae=transformer.vae,
        conditioner=transformer.conditioner,
        transformer=transformer.get_inference_denoiser(use_ema=True),
        scheduler=scheduler,
    ).to(args.device)

    def generate(prompt, num_images, seed, image_height, image_width, num_steps, guidance, timeshift, order):
        images = pipeline(
            prompt=prompt,
            num_images_per_prompt=int(num_images),
            generator=torch.Generator(device="cpu").manual_seed(int(seed)),
            height=int(image_height),
            width=int(image_width),
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance),
            timeshift=float(timeshift),
            order=int(order),
            output_type="pil",
        ).images
        return images

    with gr.Blocks() as demo:
        gr.Markdown(f"config:{args.config}\n\n ckpt_path:{args.ckpt_path}")
        with gr.Row():
            with gr.Column(scale=1):
                num_steps = gr.Slider(minimum=1, maximum=100, step=1, label="num steps", value=25)
                guidance = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, label="CFG", value=4.0)
                image_height = gr.Slider(minimum=128, maximum=1024, step=32, label="image height", value=512)
                image_width = gr.Slider(minimum=128, maximum=1024, step=32, label="image width", value=512)
                num_images = gr.Slider(minimum=1, maximum=4, step=1, label="num images", value=4)
                label = gr.Textbox(label="positive prompt", value="a photo of a cat")
                seed = gr.Slider(minimum=0, maximum=1000000, step=1, label="seed", value=0)
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