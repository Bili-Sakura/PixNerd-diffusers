# PixNerd Diffusers

Diffusers-style implementation of [PixNerd](https://arxiv.org/abs/2507.23268) (Pixel Neural Field Diffusion), following the layout of [NiT-diffusers](https://github.com/Bili-Sakura/NiT-diffusers).

Legacy training stacks, spec-based model loaders, and bundled checkpoint pipelines have been removed. Inference uses native Diffusers components under `src/diffusers`.

## Package layout

- `src/diffusers/models/transformers/transformer_pixnerd.py` — `PixNerdTransformer2DModel` and `PixNerDiT`
- `src/diffusers/models/autoencoders/autoencoder_pixel.py` — `PixNerdPixelVAE`
- `src/diffusers/models/conditioners/conditioner_pixnerd.py` — `PixNerdLabelConditioner`
- `src/diffusers/schedulers/scheduling_flow_match_pixnerd.py` — `PixNerdFlowMatchScheduler`
- `src/diffusers/pipelines/pixnerd/pipeline_pixnerd.py` — `PixNerdPipeline`
- `scripts/convert_pixnerd_ckpt_to_diffusers.py` — convert raw `.ckpt` files to a Diffusers pipeline directory
- `scripts/sample_pixnerd.py` — sample from a converted pipeline

## Install

```bash
pip install -e .
```

Or install dependencies only:

```bash
pip install -r requirements.txt
```

## Sample from CLI

```bash
python main.py sample \
  --pretrained_model_name_or_path path/to/converted-checkpoint \
  --class_label 207 \
  --num_images_per_prompt 4 \
  --output_dir samples
```

## Sample script

```bash
python scripts/sample_pixnerd.py \
  --model path/to/converted-checkpoint \
  --class-label 207 \
  --height 512 \
  --width 512 \
  --num-inference-steps 25 \
  --guidance-scale 4.0 \
  --timeshift 3.0 \
  --order 2
```

## Convert raw ImageNet checkpoints

```bash
python scripts/convert_pixnerd_ckpt_to_diffusers.py \
  --checkpoint raw/imagenet256/epoch%3D319-step%3D1600000_emainit.ckpt \
  --output models/BiliSakura/PixNerd-diffusers/PixNerd-XL-16-256
```

Batch conversion:

```bash
python scripts/convert_raw_imagenet_ckpts.py
```

Converted directories contain `model_index.json`, separate `transformer/`, `scheduler/`, `vae/`, and `conditioner/` subfolders compatible with `PixNerdPipeline.from_pretrained`.

## Python API

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))
from diffusers import PixNerdPipeline

pipe = PixNerdPipeline.from_pretrained(
    "models/BiliSakura/PixNerd-diffusers/PixNerd-XL-16-512",
    torch_dtype=torch.bfloat16,
)
images = pipe(class_labels="golden retriever", num_inference_steps=25, guidance_scale=4.0).images
```

## Gradio demo

```bash
python app.py --pretrained_model_name_or_path path/to/converted-checkpoint
```

## Upstreaming to Diffusers

Copy the files under `src/diffusers` into the matching locations in the `huggingface/diffusers` repository and register the classes in Diffusers' lazy import tables.

## Reference

```bibtex
@article{2507.23268,
  Author = {Shuai Wang and Ziteng Gao and Chenhui Zhu and Weilin Huang and Limin Wang},
  Title = {PixNerd: Pixel Neural Field Diffusion},
  Year = {2025},
  Eprint = {arXiv:2507.23268},
}
```
