"""
chatmof.moftransformer_api
==========================
Factory for the active MOFTransformer backend.

Backend selection order (first match wins):
  1. Explicit ``backend`` argument to ``get_api()``
  2. ``CHATMOF_MOFTRANSFORMER_BACKEND`` environment variable
  3. ``config['moftransformer_backend']`` in chatmof config
  4. Default: ``'stub'``

Available backends:
  'stub'   — StubMOFTransformerAPI:   no ML dependencies, returns random data
  'local'  — LocalMOFTransformerAPI:  drives an installed moftransformer package
  'remote' — RemoteMOFTransformerAPI: calls a running server.py instance over HTTP
"""

import os
from chatmof.moftransformer_api.interface import MOFTransformerAPI

_instance: MOFTransformerAPI = None


def get_api(backend: str = None) -> MOFTransformerAPI:
    """Return (and cache) the active MOFTransformer API instance."""
    global _instance
    if _instance is not None:
        return _instance

    if backend is None:
        backend = os.environ.get('CHATMOF_MOFTRANSFORMER_BACKEND')
    if backend is None:
        from chatmof.config import config
        backend = config.get('moftransformer_backend', 'stub')

    if backend == 'stub':
        from chatmof.moftransformer_api.stub import StubMOFTransformerAPI
        _instance = StubMOFTransformerAPI()
    elif backend == 'local':
        from chatmof.moftransformer_api.local import LocalMOFTransformerAPI
        _instance = LocalMOFTransformerAPI()
    elif backend == 'remote':
        from chatmof.moftransformer_api.remote import RemoteMOFTransformerAPI
        _instance = RemoteMOFTransformerAPI()
    else:
        raise ValueError(
            f"Unknown moftransformer backend: {backend!r}. "
            "Supported values: 'stub', 'local', 'remote'."
        )

    return _instance


def reset_api() -> None:
    """Clear the cached instance (useful in tests to swap backends mid-session)."""
    global _instance
    _instance = None


__all__ = ['get_api', 'reset_api', 'MOFTransformerAPI']
