from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List


class MOFTransformerAPI(ABC):
    """
    Contract between ChatMOF and a MOFTransformer backend.

    Two backends ship out of the box:
      - 'stub'  : StubMOFTransformerAPI  — no dependencies, returns dummy data
      - 'local' : LocalMOFTransformerAPI — wraps an installed moftransformer package

    Select via CHATMOF_MOFTRANSFORMER_BACKEND env-var or config['moftransformer_backend'].
    """

    @abstractmethod
    def predict(
        self,
        data_list: List[Path],
        model_dir: Path,
        verbose: bool = False,
    ) -> Dict[str, List[Any]]:
        """
        Predict a MOF property for every structure in data_list.

        Returns a dict containing:
            'cif_id'                    : List[str]   — material identifiers
            'regression_logits'         : List[float] — regression tasks
          OR
            'classification_logits_index': List[int]  — classification tasks
                (caller resolves indices via {model_dir}/label.json)
        """

    @abstractmethod
    def prepare_data(
        self,
        cif_path: Path,
        save_dir: Path,
        logger=None,
        eg_logger=None,
    ) -> bool:
        """
        Validate and featurise a generated CIF file.
        Returns True on success, False if the structure should be discarded.
        """

    @abstractmethod
    def get_predictable_properties(self) -> List[str]:
        """Return the list of property names this backend can predict."""
