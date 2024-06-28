"""Main training script."""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from cliport import agents
from cliport.dataset import RavensDataset, RavensMultiTaskDataset
from cliport.utils import utils

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


def collate_fn(batch):
    if isinstance(batch[0], dict):
        return {key: collate_fn([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], list):
        return [collate_fn([d[i] for d in batch]) for i in range(len(batch[0]))]
    elif isinstance(batch[0], tuple):
        return tuple(collate_fn([d[i] for d in batch]) for i in range(len(batch[0])))
    elif isinstance(batch[0], torch.Tensor):
        return torch.stack(batch, dim=0)
    else:
        return batch


@hydra.main(config_path="./cfg", config_name='train')
def main(cfg):
    # Set seed
    utils.set_seed(cfg['train']['seed'], torch=True)

    # Logger
    wandb_logger = WandbLogger(
        name=cfg['tag'],
        project=cfg['wandb']['logger']['project']
    ) if cfg['train']['log'] else None

    # Checkpoint saver
    hydra_dir = Path(os.getcwd())
    print(f"Hydra dir: {hydra_dir}")
    checkpoint_path = os.path.join(cfg['train']['train_dir'], 'checkpoints')
    last_checkpoint_path = os.path.join(checkpoint_path, 'last.ckpt')
    last_checkpoint = last_checkpoint_path if (
            os.path.exists(last_checkpoint_path) and cfg['train']['load_from_last_ckpt']
    ) else None
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg['wandb']['saver']['monitor'],
        dirpath=checkpoint_path,
        filename='best',
        save_top_k=1,
        save_last=True,
    )

    # Trainer
    max_epochs = cfg['train']['n_steps'] // cfg['train']['n_demos']
    print(f"N demos: {cfg['train']['n_demos']}")
    print(f"Max epochs: {max_epochs}")
    print(f"config: {cfg}")
    trainer = Trainer(
        devices=cfg['train']['gpu'],
        accelerator='gpu',
        fast_dev_run=cfg['debug'],
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        max_epochs=max_epochs,
        check_val_every_n_epoch=min(max_epochs // 5, 10),
        log_every_n_steps=min(max_epochs // 10, 10),
    )

    # Resume epoch and global_steps
    # if last_checkpoint:
    #     print(f"Resuming: {last_checkpoint}")
    #     last_ckpt = torch.load(last_checkpoint)
    #     trainer
    #     trainer.current_epoch = 9
    #     trainer.global_step = 16669
    #     # trainer.current_epoch = last_ckpt['epoch']
    #     # trainer.global_step = last_ckpt['global_step']
    #     del last_ckpt

    # Config
    data_dir = cfg['train']['data_dir']
    task = cfg['train']['task']
    task_difficulty_level = cfg['train']['task_difficulty_level']
    if task_difficulty_level == 'hard':
        task += '-hard'
    agent_type = cfg['train']['agent']
    n_demos = cfg['train']['n_demos']
    n_val = cfg['train']['n_val']
    batch_size = cfg['train']['batch_size']
    name = '{}-{}-{}'.format(task, agent_type, n_demos)

    # Datasets
    dataset_type = cfg['dataset']['type']
    if 'multi' in dataset_type:
        train_ds = RavensMultiTaskDataset(
            data_dir, cfg, group=task, mode='train', n_demos=n_demos, augment=True
        )
        val_ds = RavensMultiTaskDataset(
            data_dir, cfg, group=task, mode='val', n_demos=n_val, augment=False
        )
    # elif 'real' in dataset_type:
    #     train_ds = RealDataset(
    #         os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=False
    #     )
    #     val_ds = RealDataset(
    #         os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False
    #     )
    else:
        train_ds = RavensDataset(
            os.path.join(data_dir, '{}-train'.format(task)), cfg, n_demos=n_demos, augment=True
        )
        val_ds = RavensDataset(
            os.path.join(data_dir, '{}-val'.format(task)), cfg, n_demos=n_val, augment=False
        )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, collate_fn=collate_fn, num_workers=32
    )
    val_loader = DataLoader(
        val_ds, collate_fn=collate_fn, num_workers=8
    )

    # Initialize agent
    agent = agents.names[agent_type](name, cfg, train_loader, val_loader)
    agent.automatic_optimization = False
    # print(agent)

    # Main training loop
    trainer.fit(agent, ckpt_path=last_checkpoint)


if __name__ == '__main__':
    main()
