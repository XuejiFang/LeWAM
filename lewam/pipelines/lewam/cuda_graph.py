"""Bounded, fixed-shape CUDA Graph storage for stateless policy requests."""

import torch


class InferenceGraph:
    """Record a complete request and copy fresh inputs before every replay."""

    @torch.no_grad()
    def __init__(self, forward, inputs, resources):
        # Graphs hold device addresses, not Python ownership of pre-capture tensors.
        # Keep masks, RoPE constants and scheduler tensors alive across replays.
        self.resources = resources
        self.inputs = tuple(value.clone() for value in inputs)
        self.device = self.inputs[0].device
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                outputs = forward(*self.inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        del outputs
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = forward(*self.inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)

    @torch.no_grad()
    def replay(self, inputs, *, return_state=False):
        for source, target in zip(inputs, self.inputs):
            target.copy_(source)
        self.graph.replay()
        # Independent storage prevents the next observation overwriting this result.
        action = self.outputs[0].clone()
        state = self.outputs[1].clone() if return_state else None
        return action, state

    def close(self):
        # Release only after outstanding replays have finished on their CUDA stream.
        torch.cuda.synchronize(self.device)
        self.graph.reset()
        self.inputs = ()
        self.outputs = ()
        self.resources = ()


def model_storage_signature(module):
    """In-place weight updates stay valid; storage replacement requires recapture."""
    parameters = tuple((p.data_ptr(), p.device, p.dtype) for p in module.parameters())
    buffers = tuple(
        (p.data_ptr(), p.device, p.dtype, p._version) for p in module.buffers()
    )
    return id(module), parameters, buffers
