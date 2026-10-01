"""Optional PyTorch / PEFT bridge (install with `pip install -e ".[torch]"`).

* TorchLoRAClassifier is a drop-in for model.LoRAClassifier whose gradients come from autograd
  instead of hand-written numpy; the federation code is unchanged.
* to_peft / save_peft_adapter / load_peft_adapter map our (A, B) factors onto Hugging Face PEFT
  LoRA layers (same alpha/r scaling), so an adapter trained here can be loaded by PEFT and the
  other way round.

What this does NOT do: run bitsandbytes 4-bit kernels or an 8B LLM. The 4-bit base is the same
simulated NF4 round-trip as the numpy path, and no GPU or model weights are involved.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from fedlora.model import Adapter, LoRAClassifier

PEFT_MODULE = "base"  # attribute name of the adapted layer inside the toy network


class TorchLoRAClassifier(LoRAClassifier):
    """LoRAClassifier with autograd gradients (float64, so it matches numpy to ~1e-12)."""

    def grads(self, x, y, train: Adapter, *frozen: Adapter) -> tuple[np.ndarray, np.ndarray]:
        dt = torch.float64
        xt = torch.as_tensor(x, dtype=dt)
        z = xt @ torch.as_tensor(self.W, dtype=dt).T + torch.as_tensor(self.b, dtype=dt)
        for ad in frozen:
            z = z + self.scale * (xt @ torch.as_tensor(ad.A, dtype=dt).T) @ torch.as_tensor(ad.B, dtype=dt).T
        A = torch.tensor(train.A, dtype=dt, requires_grad=True)
        B = torch.tensor(train.B, dtype=dt, requires_grad=True)
        z = z + self.scale * (xt @ A.T) @ B.T
        loss = nn.functional.cross_entropy(z, torch.as_tensor(y, dtype=torch.long))
        loss.backward()
        return A.grad.numpy(), B.grad.numpy()


class _Net(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.base = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.base(x)


def to_peft(model: LoRAClassifier, adapter: Adapter, alpha: float | None = None):
    """A PEFT-wrapped torch module computing exactly model.logits(x, adapter)."""
    from peft import LoraConfig, get_peft_model

    r, d_in = adapter.A.shape
    d_out = adapter.B.shape[0]
    alpha = alpha if alpha is not None else model.scale * r
    net = _Net(d_in, d_out).double()
    with torch.no_grad():
        net.base.weight.copy_(torch.as_tensor(model.W))
        net.base.bias.copy_(torch.as_tensor(model.b))
    cfg = LoraConfig(r=r, lora_alpha=alpha, target_modules=[PEFT_MODULE], lora_dropout=0.0,
                     bias="none", init_lora_weights=False)
    peft_net = get_peft_model(net, cfg).double()
    layer = peft_net.base_model.model.base
    with torch.no_grad():
        layer.lora_A["default"].weight.copy_(torch.as_tensor(adapter.A))
        layer.lora_B["default"].weight.copy_(torch.as_tensor(adapter.B))
    return peft_net


def from_peft(peft_net) -> Adapter:
    layer = peft_net.base_model.model.base
    return Adapter(layer.lora_A["default"].weight.detach().numpy().copy(),
                   layer.lora_B["default"].weight.detach().numpy().copy())


def save_peft_adapter(model: LoRAClassifier, adapter: Adapter, directory: Path) -> None:
    """Write adapter_config.json + adapter_model.safetensors in the standard PEFT layout."""
    to_peft(model, adapter).save_pretrained(str(directory))


def load_peft_adapter(model: LoRAClassifier, directory: Path) -> Adapter:
    from peft import PeftModel

    d_out, d_in = model.W.shape
    net = _Net(d_in, d_out).double()
    return from_peft(PeftModel.from_pretrained(net, str(directory)).double())
