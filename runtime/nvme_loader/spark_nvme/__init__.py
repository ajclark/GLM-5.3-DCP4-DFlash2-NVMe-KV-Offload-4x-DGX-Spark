"""Importing transport/artifacts does not import vLLM or initialize CUDA."""

def register():
    from .loader import register_loader
    register_loader()
