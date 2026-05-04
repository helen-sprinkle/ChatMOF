"""
ChatDataset — local backend only.

This file is intentionally NOT imported at the package level so that
importing chatmof.moftransformer_api never triggers a moftransformer
or torch import.  LocalMOFTransformerAPI loads it on demand.
"""
import torch
from pathlib import Path
from typing import Dict, List

from moftransformer.datamodules import Dataset as MOFDataset


class ChatDataset(MOFDataset):
    """
    Minimal inference-only dataset backed by a list of CIF file paths.

    Skips MOFDataset.__init__ (which expects a data-module config) and
    calls torch.utils.data.Dataset.__init__ directly, setting only the
    attributes that MOFDataset.__getitem__ and collate actually read.
    """

    def __init__(
        self,
        data_list: List[Path],
        nbr_fea_len: int = 64,
    ) -> Dict[str, torch.Tensor]:
        super(MOFDataset, self).__init__()
        self.data_list = data_list
        self.draw_false_grid = False
        self.split = ''
        self.nbr_fea_len = nbr_fea_len
        self.tasks = {}
        self.cif_ids = [cif.stem for cif in data_list]
        self.data_dir = data_list[0].parent
        self.targets = [0] * len(self.cif_ids)
