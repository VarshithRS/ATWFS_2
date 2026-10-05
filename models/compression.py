"""Compression utilities: INT8 post-training quantization and structured pruning.

* :func:`quantize_int8_ptq`  - FX-graph-mode *static* PTQ (calibrated on real/synthetic batches).
  Falls back to dynamic INT8 on Linear layers (clearly logged) if FX quantization is unavailable.
* :func:`prune_decoder_structured` - physically removes the lowest-L1 filters of the *first* conv of every
  decoder block (and the matching input channels of the second conv), so the model genuinely shrinks.
  Only the decoder is pruned (the MobileNet encoder has depthwise/SE/residual couplings).
* :func:`model_size_mb`, :func:`cpu_fps` - benchmarking helpers.
"""
from __future__ import annotations

import copy
import io
import logging
import time
from typing import Any, Dict, Iterable, Tuple

import torch
import torch.nn as nn

from models.unet_mobilenet import ConvGNAct, MultiTaskUNet, gn_groups

logger = logging.getLogger(__name__)


class _TupleOut(nn.Module):
    """FX-traceable wrapper: model -> (seg_logits, alpha, terrain_logits) tuple."""

    def __init__(self, model: MultiTaskUNet) -> None:
        """Wrap a model.

        Args:
            model: Float model.
        """
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward returning a tuple.

        Args:
            x: Input batch.

        Returns:
            ``(seg_logits, alpha, terrain_logits)``.
        """
        o = self.model(x)
        return o["seg_logits"], o["alpha"], o["terrain_logits"]


class DictOutput(nn.Module):
    """Re-wraps a tuple-returning (quantized) module to the standard output dict."""

    def __init__(self, inner: nn.Module) -> None:
        """Wrap a module.

        Args:
            inner: Tuple-returning module.
        """
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Forward returning the standard dict (``aux_logits`` is empty for deployed models).

        Args:
            x: Input batch.

        Returns:
            Output dict.
        """
        seg, alpha, terr = self.inner(x)
        return {"seg_logits": seg, "alpha": alpha.clamp(min=1.0), "terrain_logits": terr, "aux_logits": []}


def _pick_engine() -> str:
    """Select the best available quantized CPU engine.

    Returns:
        Engine name.
    """
    sup = torch.backends.quantized.supported_engines
    for e in ("x86", "fbgemm", "qnnpack"):
        if e in sup:
            return e
    return torch.backends.quantized.engine


def quantize_int8_ptq(model: MultiTaskUNet, calib_batches: Iterable[torch.Tensor], example: torch.Tensor
                      ) -> Tuple[nn.Module, str]:
    """INT8 post-training quantization (CPU).

    Args:
        model: Float model (will not be modified).
        calib_batches: Iterable of input image batches used for calibration.
        example: Example input for tracing.

    Returns:
        ``(quantized model returning the standard output dict, method description)``.
    """
    base = copy.deepcopy(model).cpu().eval()
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

        torch.backends.quantized.engine = _pick_engine()
        prepared = prepare_fx(_TupleOut(base).eval(), get_default_qconfig_mapping(torch.backends.quantized.engine),
                              (example.cpu(),))
        with torch.no_grad():
            for xb in calib_batches:
                prepared(xb.cpu())
        q = convert_fx(prepared)
        return DictOutput(q).eval(), f"int8_static_ptq_fx[{torch.backends.quantized.engine}]"
    except Exception as e:  # pragma: no cover - platform dependent
        logger.warning("FX static PTQ failed (%s: %s). Falling back to DYNAMIC INT8 on Linear layers only.",
                       type(e).__name__, e)
        try:
            dq = torch.ao.quantization.quantize_dynamic(base, {nn.Linear}, dtype=torch.qint8)
            return dq.eval(), "int8_dynamic_linear_only"
        except Exception as e2:  # pragma: no cover
            logger.error("Dynamic INT8 also unavailable (%s). Returning an UNQUANTIZED fp32 copy; "
                         "migrate this function to torchao if your torch removed torch.ao.", e2)
            return DictOutput(_TupleOut(base)).eval(), "int8_UNAVAILABLE_fp32_copy"


def prune_decoder_structured(model: MultiTaskUNet, amount: float) -> MultiTaskUNet:
    """Structured (filter-level) pruning of the decoder with real shape reduction.

    For every decoder block, ranks ``conv1`` output filters by L1 norm, keeps the top
    ``(1 - amount)`` fraction and slices ``conv1``/its GroupNorm/``conv2`` input channels accordingly.

    Args:
        model: Source model (not modified).
        amount: Fraction of internal decoder channels to remove, in [0, 1).

    Returns:
        New, smaller model.
    """
    if not 0.0 <= amount < 1.0:
        raise ValueError("amount must be in [0, 1)")
    m = copy.deepcopy(model).cpu().eval()
    for name in ("dec16", "dec8", "dec4", "dec2"):
        blk = getattr(m, name)
        conv1, gn1, conv2 = blk.conv1[0], blk.conv1[1], blk.conv2[0]
        c = conv1.out_channels
        keep = max(4, int(round(c * (1.0 - amount))))
        idx = torch.sort(torch.topk(conv1.weight.detach().abs().sum(dim=(1, 2, 3)), keep).indices).values
        n1 = nn.Conv2d(conv1.in_channels, keep, conv1.kernel_size, padding=conv1.padding, bias=False)
        n1.weight.data = conv1.weight.data[idx].clone()
        ng = nn.GroupNorm(gn_groups(keep, gn1.num_groups), keep, eps=gn1.eps)
        ng.weight.data, ng.bias.data = gn1.weight.data[idx].clone(), gn1.bias.data[idx].clone()
        n2 = nn.Conv2d(keep, conv2.out_channels, conv2.kernel_size, padding=conv2.padding, bias=False)
        n2.weight.data = conv2.weight.data[:, idx].clone()
        blk.conv1[0], blk.conv1[1], blk.conv2[0] = n1, ng, n2
        logger.info("pruned %s: %d -> %d internal channels", name, c, keep)
    return m


def model_size_mb(model: nn.Module) -> float:
    """Serialized ``state_dict`` size in MB.

    Args:
        model: Any module (float, pruned or quantized).

    Returns:
        Size in megabytes (1 MB = 1e6 bytes).
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1e6


@torch.no_grad()
def cpu_fps(model: nn.Module, input_shape: Tuple[int, int, int, int], warmup: int = 3, iters: int = 10) -> float:
    """Batch-1 CPU inference throughput.

    Args:
        model: Model (moved to CPU, eval).
        input_shape: ``(1,3,H,W)``.
        warmup: Warm-up iterations.
        iters: Timed iterations.

    Returns:
        Frames per second.
    """
    model = model.cpu().eval()
    x = torch.randn(*input_shape)
    for _ in range(warmup):
        model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    return iters / (time.perf_counter() - t0)
