# ChatDataset has moved to chatmof.moftransformer_api._dataset (local backend only).
# This shim preserves import compatibility for code that referenced the old path.
try:
    from chatmof.moftransformer_api._dataset import ChatDataset
    __all__ = ['ChatDataset']
except ImportError:
    pass
