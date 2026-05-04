import random
from pathlib import Path
from typing import Any, Dict, List

from chatmof.moftransformer_api.interface import MOFTransformerAPI

# Properties known to be used in ChatMOF prompts and examples.
# The LLM uses this list to know what it can ask the predictor tool for.
STUB_PROPERTIES: List[str] = [
    'bandgap',
    'void_fraction',
    'surface_area',
    'pore_volume',
    'hydrogen_uptake',
    'solvent_removal_stability',
    'largest_free_pore_diameter',
    'pore_limiting_diameter',
]


class StubMOFTransformerAPI(MOFTransformerAPI):
    """
    Dummy backend for development and testing.

    Returns random float predictions so that ChatMOF's full agent loop
    (LLM → tool dispatch → result formatting) can run without moftransformer
    or any ML dependencies installed.
    """

    def predict(
        self,
        data_list: List[Path],
        model_dir: Path,
        verbose: bool = False,
    ) -> Dict[str, List[Any]]:
        cif_ids = [p.stem for p in data_list]
        logits = [round(random.uniform(0.0, 5.0), 4) for _ in cif_ids]
        return {'cif_id': cif_ids, 'regression_logits': logits}

    def prepare_data(
        self,
        cif_path: Path,
        save_dir: Path,
        logger=None,
        eg_logger=None,
    ) -> bool:
        return True

    def get_predictable_properties(self) -> List[str]:
        return list(STUB_PROPERTIES)
