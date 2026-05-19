# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Any


def _is_local_diffusers_module(module_name: str) -> bool:
    module = sys.modules.get(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return False
    normalized = module_file.replace("\\", "/")
    return "/src/diffusers" in normalized


def _local_diffusers_module_names() -> list[str]:
    return [
        name
        for name in list(sys.modules)
        if name == "diffusers" or name.startswith("diffusers.")
        if _is_local_diffusers_module(name)
    ]


def import_hf_diffusers_module(relative_path: str) -> ModuleType:
    """
    Import a module from the installed Hugging Face `diffusers` distribution.

    The local `src/diffusers` staging package shadows the PyPI package on `sys.path`,
    so this helper temporarily unloads only the local package before importing.
    """
    module_name = f"diffusers.{relative_path}"
    src_paths = [path for path in sys.path if path.endswith("/src") or path.endswith("\\src")]
    removed_paths = [path for path in src_paths if path in sys.path]
    for path in removed_paths:
        sys.path.remove(path)

    local_module_names = _local_diffusers_module_names()
    cached_modules = {name: sys.modules.pop(name) for name in local_module_names}
    try:
        return importlib.import_module(module_name)
    finally:
        sys.modules.update(cached_modules)
        for path in reversed(removed_paths):
            sys.path.insert(0, path)


def get_hf_diffusers_attr(relative_path: str, attr: str) -> Any:
    module = import_hf_diffusers_module(relative_path)
    return getattr(module, attr)
