from pathlib import Path
from typing import Dict, List

from chatmof.moftransformer_api import get_api

_predictable_properties: List[str] = get_api().get_predictable_properties()
model_names: str = ",".join(_predictable_properties + ['else'])


def predict(
        data_list: List[Path],
        model_dir: Path,
        verbose: bool = False,
) -> Dict[str, List]:
    return get_api().predict(data_list, model_dir, verbose=verbose)


def search_file(name: str, direc: Path) -> List[Path]:
    name = name.strip()
    if '*' in name:
        return list(direc.glob(name))
    f_name = direc / name
    if f_name.exists():
        return [f_name]
    return False
