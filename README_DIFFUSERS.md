PixNerd Diffusers integration
=============================

Native Diffusers components live under `src/diffusers`:

- `models/transformers/transformer_pixnerd.py` — `PixNerdTransformer2DModel`
- `models/autoencoders/autoencoder_pixel.py` — `PixNerdPixelVAE`
- `models/conditioners/conditioner_pixnerd.py` — `PixNerdLabelConditioner`
- `schedulers/scheduling_flow_match_pixnerd.py` — `PixNerdFlowMatchScheduler`
- `pipelines/pixnerd/pipeline_pixnerd.py` — `PixNerdPipeline`

Install and sample:

```bash
pip install -e .

python scripts/sample_pixnerd.py \
  --model path/to/converted-checkpoint \
  --class-label 207 \
  --num-inference-steps 25 \
  --guidance-scale 4.0
```

Convert a raw checkpoint:

```bash
python scripts/convert_pixnerd_ckpt_to_diffusers.py \
  --checkpoint raw/imagenet256/epoch%3D319-step%3D1600000_emainit.ckpt \
  --output pretrained_models/BiliSakura/PixNerd-diffusers/PixNerd-XL-16-256
```
