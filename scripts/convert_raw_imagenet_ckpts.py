#!/usr/bin/env python3

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_pixnerd_ckpt_to_diffusers import convert_checkpoint


def main() -> None:
    repo_root = REPO_ROOT
    output_root = repo_root / "pretrained_models" / "BiliSakura" / "PixNerd-diffusers"

    jobs = [
        {
            "checkpoint": repo_root / "raw" / "imagenet256" / "epoch%3D319-step%3D1600000_emainit.ckpt",
            "output": output_root / "PixNerd-XL-16-256",
            "num_inference_steps": 100,
            "guidance_scale": 3.5,
            "timeshift": 3.0,
            "order": 2,
            "use_ema": True,
        },
        {
            "checkpoint": repo_root / "raw" / "imagenet512" / "res512_ft200k_epoch%3D325-step%3D1800000_emainit.ckpt",
            "output": output_root / "PixNerd-XL-16-512",
            "num_inference_steps": 100,
            "guidance_scale": 3.5,
            "timeshift": 3.0,
            "order": 2,
            "use_ema": True,
        },
    ]

    for job in jobs:
        checkpoint = Path(job["checkpoint"])
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        print(f"Converting {checkpoint} -> {job['output']}")
        convert_checkpoint(
            checkpoint_path=checkpoint,
            output_dir=Path(job["output"]),
            num_inference_steps=job["num_inference_steps"],
            guidance_scale=job["guidance_scale"],
            timeshift=job["timeshift"],
            order=job["order"],
            use_ema=job["use_ema"],
        )

    print(f"Done. Converted checkpoints saved under: {output_root}")


if __name__ == "__main__":
    main()
