import torch
from src.models.conditioner.base import BaseConditioner, resolve_conditioner_device

class LabelConditioner(BaseConditioner):
    def __init__(self, num_classes):
        super().__init__()
        self.null_condition = num_classes

    def _impl_condition(self, y, metadata):
        device = resolve_conditioner_device(metadata)
        return torch.tensor(y, device=device).long()

    def _impl_uncondition(self, y, metadata):
        device = resolve_conditioner_device(metadata)
        return torch.full((len(y),), self.null_condition, dtype=torch.long, device=device)