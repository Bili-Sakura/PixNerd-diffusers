# PixNerd Diffusers Refactor

This repository is refactored to a Diffusers-first inference flow inspired by
[NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers):

- legacy `src/diffusion/*` training/sampling stack removed
- runtime entrypoints use `diffusers.DiffusionPipeline.from_pretrained(...)`
- PixNerd custom model/scheduler/pipeline stay in `src/pixnerd_diffusers`

## Install

```bash
pip install -r requirements.txt
```

## Sample from CLI

```bash
python main.py sample \
  --pretrained_model_name_or_path MCG-NJU/PixNerd-XXL-P16-T2I \
  --prompt "a photo of a cat" \
  --num_images_per_prompt 4 \
  --output_dir samples
```

Class-conditional usage:

```bash
python main.py sample \
  --pretrained_model_name_or_path path/to/checkpoint \
  --class_label 207 \
  --num_images_per_prompt 4 \
  --output_dir samples
```

## Sample script (NiT-style)

```bash
python scripts/sample_pixnerd.py \
  --model MCG-NJU/PixNerd-XXL-P16-T2I \
  --prompt "a photo of a cat" \
  --height 512 \
  --width 512 \
  --num-inference-steps 25 \
  --guidance-scale 4.0 \
  --timeshift 3.0 \
  --order 2
```

## Convert raw ImageNet checkpoints

Batch conversion (both ImageNet256 and ImageNet512):

```bash
conda activate sakura
python scripts/convert_raw_imagenet_ckpts.py
```

This reads:

- `raw/imagenet256/epoch%3D319-step%3D1600000_emainit.ckpt`
- `raw/imagenet512/res512_ft200k_epoch%3D325-step%3D1800000_emainit.ckpt`

and writes Diffusers-style checkpoints to:

- `pretrained_models/BiliSakura/PixNerd-diffusers/PixNerd-XL-16-256`
- `pretrained_models/BiliSakura/PixNerd-diffusers/PixNerd-XL-16-512`

Each converted checkpoint directory is now self-contained and includes:

- `pipeline.py` (single bundled custom pipeline and all needed classes)

So the checkpoint repo only needs `diffusers` + `torch` at runtime.

Single-checkpoint conversion:

```bash
conda activate sakura
python scripts/convert_pixnerd_ckpt_to_diffusers.py \
  --checkpoint raw/imagenet256/epoch%3D319-step%3D1600000_emainit.ckpt \
  --output pretrained_models/BiliSakura/PixNerd-diffusers/PixNerd-XL-16-256
```

## Python API

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "path/to/checkpoint",
    custom_pipeline="path/to/checkpoint/pipeline.py",
)
images = pipe(prompt="a photo of a cat", num_inference_steps=25, guidance_scale=4.0).images
```

## Gradio demo

```bash
python app.py --pretrained_model_name_or_path MCG-NJU/PixNerd-XXL-P16-T2I
```

## Reference

```bibtex
@article{2507.23268,
Author = {Shuai Wang and Ziteng Gao and Chenhui Zhu and Weilin Huang and Limin Wang},
Title = {PixNerd: Pixel Neural Field Diffusion},
Year = {2025},
Eprint = {arXiv:2507.23268},
}
```
