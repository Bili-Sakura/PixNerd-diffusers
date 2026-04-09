from __future__ import annotations

import importlib
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def to_container(config: Any) -> Any:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return config


def load_symbol(path: str) -> Any:
    module_path, name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, name)


def instantiate_from_spec(spec: Any) -> Any:
    spec = to_container(spec)
    if isinstance(spec, dict) and "class_path" in spec:
        class_or_fn = load_symbol(spec["class_path"])
        init_args = spec.get("init_args", {})
        if isinstance(init_args, dict):
            init_args = {k: instantiate_from_spec(v) for k, v in init_args.items()}
        return class_or_fn(**init_args)
    if isinstance(spec, dict):
        return {k: instantiate_from_spec(v) for k, v in spec.items()}
    if isinstance(spec, list):
        return [instantiate_from_spec(v) for v in spec]
    if isinstance(spec, str) and "." in spec:
        try:
            return load_symbol(spec)
        except Exception:
            return spec
    return spec


def clone_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    return to_container(OmegaConf.create(to_container(spec)))
