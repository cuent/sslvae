"""
Trainer of SSLVAE-M2
"""
import hydra

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from src.sslvae.sslvae import SSLVAE
from pytorch_lightning.loggers import TensorBoardLogger


@hydra.main(config_path='conf', config_name='config')
def main(cfg: DictConfig):
    model = SSLVAE(cfg)

    # fast_dev_run=True # run train and val
    logger = TensorBoardLogger('tb_logs', name='my_model')
    trainer = Trainer(
        max_epochs=cfg.train.max_epochs,
        gpus=cfg.train.gpus,
        logger=logger,
        # fast_dev_run=True,
    )
    trainer.fit(model)


if __name__ == '__main__':
    main()
