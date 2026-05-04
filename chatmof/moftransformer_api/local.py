"""
LocalMOFTransformerAPI — calls moftransformer installed in the local environment.

All moftransformer / torch / pytorch_lightning imports are deferred to method
bodies so that importing this module never fails even if those packages are absent.
The ImportError surfaces only when predict() or prepare_data() is actually called.
"""
import logging
import yaml
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List

from chatmof.config import config as chatmof_config
from chatmof.moftransformer_api.interface import MOFTransformerAPI


class LocalMOFTransformerAPI(MOFTransformerAPI):
    """
    Thin adapter that drives a locally-installed moftransformer package.

    Requires: moftransformer >= 2.0, pytorch_lightning, torch
    """

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_predictable_properties(self) -> List[str]:
        model_dir = Path(chatmof_config['model_dir'])
        if not model_dir.exists():
            return []
        return [p.stem for p in model_dir.iterdir() if p.is_dir()]

    def predict(
        self,
        data_list: List[Path],
        model_dir: Path,
        verbose: bool = False,
    ) -> Dict[str, List[Any]]:
        import pytorch_lightning as pl
        from torch.utils.data import DataLoader
        from moftransformer.modules import Module
        from moftransformer.utils.validation import _IS_INTERACTIVE
        from chatmof.moftransformer_api._dataset import ChatDataset

        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        cfg = self._load_hparams(model_dir)
        pl.seed_everything(cfg['seed'])
        cfg.update({
            'accelerator': chatmof_config['accelerator'],
            'load_path': str(self._find_checkpoint(model_dir)),
            'max_epochs': 1,
            'devices': 1,
        })

        dataloader = DataLoader(
            dataset=ChatDataset(data_list=data_list, nbr_fea_len=cfg['nbr_fea_len']),
            collate_fn=partial(ChatDataset.collate, img_size=cfg['img_size']),
            batch_size=cfg['per_gpu_batchsize'],
            num_workers=cfg['num_workers'],
        )

        trainer = pl.Trainer(
            accelerator=cfg['accelerator'],
            devices=cfg['devices'],
            num_nodes=cfg['num_nodes'],
            precision=cfg['precision'],
            strategy=self._ddp_strategy(pl, _IS_INTERACTIVE),
            max_epochs=cfg['max_epochs'],
            log_every_n_steps=0,
            logger=False,
        )

        rets = trainer.predict(Module(cfg), dataloader)
        output: Dict[str, list] = defaultdict(list)
        for ret in rets:
            for key, value in ret.items():
                output[key].extend(value)
        return dict(output)

    def prepare_data(
        self,
        cif_path: Path,
        save_dir: Path,
        logger=None,
        eg_logger=None,
    ) -> bool:
        from moftransformer.utils.prepare_data import make_prepared_data
        return make_prepared_data(cif_path, save_dir, logger=logger, eg_logger=eg_logger)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _load_hparams(model_dir: Path) -> Dict[str, Any]:
        params = model_dir / 'hparams.yaml'
        if not params.exists():
            raise FileNotFoundError(f'hparams.yaml not found in {model_dir}')
        with open(params) as f:
            return yaml.load(f, Loader=yaml.Loader)['config']

    @staticmethod
    def _find_checkpoint(model_dir: Path) -> Path:
        ckpts = list(model_dir.glob('*.ckpt'))
        if not ckpts:
            raise FileNotFoundError(f'No .ckpt file found in {model_dir}')
        return ckpts[0]

    @staticmethod
    def _ddp_strategy(pl, is_interactive: bool):
        if is_interactive:
            return None
        if pl.__version__ >= '2.0.0':
            return 'ddp_find_unused_parameters_true'
        return 'ddp'
