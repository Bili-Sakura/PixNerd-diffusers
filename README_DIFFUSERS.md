PixNerd Diffusers integration
=============================

This repository now keeps only a Diffusers-first inference path.

- `src/pixnerd_diffusers/models/modeling_pixnerd_transformer_2d.py` defines
  `PixNerdTransformer2DModel` (`ModelMixin` + `ConfigMixin`).
- `src/pixnerd_diffusers/schedulers/scheduling_pixnerd_flow_match.py` defines
  `PixNerdFlowMatchScheduler`.
- `src/pixnerd_diffusers/pipelines/pipeline_pixnerd.py` defines
  `PixNerdPipeline`.
- `scripts/sample_pixnerd.py` is a standalone sample script.

Sample
------

```bash
python scripts/sample_pixnerd.py \
  --model MCG-NJU/PixNerd-XXL-P16-T2I \
  --prompt "a photo of a cat" \
  --num-inference-steps 25 \
  --guidance-scale 4.0
```

For CLI usage through the repository entrypoint:

```bash
python main.py sample \
  --pretrained_model_name_or_path MCG-NJU/PixNerd-XXL-P16-T2I \
  --prompt "a photo of a cat"
```
