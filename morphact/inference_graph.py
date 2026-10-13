"""Bounded CUDA Graph replay for a fixed inference function."""

from __future__ import annotations

import logging

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

logger = logging.getLogger(__name__)


class CUDAGraphRunner:
    """Capture deterministic denoising on a fixed eval model, with Tensor output.

    Calls must be serial on one CUDA stream. Supply noise and every changing
    condition in args; sampling and parameter generation belong outside the
    function. Rebuild the runner after loading weights or changing the model.
    CPU or mixed-device tensor inputs run eagerly. Unsupported captures occupy
    a cache slot, so both successful and failed signatures remain bounded.
    """

    def __init__(self, function, max_graphs=4):
        if not isinstance(max_graphs, int) or max_graphs <= 0:
            raise ValueError("max_graphs must be a positive integer")
        self.function = function
        self.max_graphs = max_graphs
        # TreeSpec contains dict key lists; equality avoids requiring a hash.
        self._entries = []

    def __call__(self, *args):
        leaves, spec = tree_flatten(args)
        tensors = [value for value in leaves if isinstance(value, torch.Tensor)]
        if not tensors or any(value.device.type != "cuda" or value.device != tensors[0].device for value in tensors):
            return self.function(*args)
        signature = (
            spec,
            # Replay bypasses autocast dispatch and retains the captured precision.
            torch.is_autocast_enabled("cuda"),
            torch.get_autocast_dtype("cuda"),
            tuple(
                (torch.Tensor, tuple(value.shape), value.dtype, value.device)
                if isinstance(value, torch.Tensor)
                else (type(value), value)
                for value in leaves
            ),
        )
        with torch.no_grad(), torch.cuda.device(tensors[0].device):
            for cached_signature, entry in self._entries:
                if cached_signature == signature:
                    break
            else:
                if len(self._entries) >= self.max_graphs:
                    return self.function(*args)
                try:
                    # Cached weight casts can be freed when the caller exits autocast.
                    with torch.autocast(
                        "cuda",
                        enabled=torch.is_autocast_enabled("cuda"),
                        dtype=torch.get_autocast_dtype("cuda"),
                        cache_enabled=False,
                    ):
                        entry = self._capture(leaves, spec, tensors[0].device)
                except RuntimeError as error:
                    message = str(error).lower()
                    if not any(
                        reason in message
                        for reason in (
                            "operation not permitted when stream is capturing",
                            "cudaerrorstreamcaptureunsupported",
                        )
                    ):
                        raise
                    logger.warning("CUDA Graph capture unsupported; this signature will run eagerly: %s", error)
                    entry = None
                self._entries.append((signature, entry))
            if entry is None:
                return self.function(*args)
            graph, buffers, output = entry
            for destination, source in zip(buffers, leaves, strict=True):
                if isinstance(destination, torch.Tensor):
                    destination.copy_(source)
            graph.replay()
            return output.clone()

    def _capture(self, leaves, spec, device):
        buffers = [value.detach().clone() if isinstance(value, torch.Tensor) else value for value in leaves]
        args = tree_unflatten(buffers, spec)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.function(*args)
        torch.cuda.current_stream(device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            graph.capture_begin()
            try:
                output = self.function(*args)
            except BaseException:
                # End invalidated capture without hiding the original model error.
                try:
                    graph.capture_end()
                except RuntimeError:
                    pass
                raise
            graph.capture_end()
        return graph, buffers, output
