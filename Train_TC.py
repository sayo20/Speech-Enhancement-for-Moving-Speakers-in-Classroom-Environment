import pytorch_lightning as pl
from pytorch_lightning import Trainer
# from option import parse
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from lightning_ipd import Lightning

import torch
import argparse
import os
from Callbacks import EarlyStopping_
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from LightningFile_TC import Lightning
# from Lightning_enhancement import Lightning
from models import SeparationNet,EnhancementNet
from pytorch_lightning.strategies import DDPStrategy

#get exp as arguement and then get lightning import and conf names..how do we pass exp to lightning
import sentry_sdk
import wandb

wandb.login(key="d39daf4c1d304b5fa7f5e1a79958bd85ef105f30")
# parser = argparse.ArgumentParser()
# parser.add_argument("-e", "--exp", help = "Experiment number")

# args = parser.parse_args()
# exp = args.exp


def Train():
    torch.set_float32_matmul_precision('medium') 
    config_name = "config_TC.yaml"
    run_name = "MIMO-Separate-ClassFinetune-Bab-ChildBRIR"

    wandb_logger = WandbLogger(project="MovingSpeakerClass", config=config_name,name=run_name)
    print("Taking off...")
    config = wandb_logger.experiment.config
 
    epoch = config["epochs"]


    early_stop_callback = EarlyStopping(monitor="val_loss", patience=2, verbose=1, mode="min")#monitor="val_loss", patience=12, verbose=1, mode="min",start_epoch=150
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",#val_loss
        dirpath=config["save_dir"],
        save_top_k = 3, 
        # save_top_k=-1,                  # Save all checkpoints
        # every_n_epochs=5,               # ✅ Save every 5 epochs
        verbose=True,
        filename=config["save_name"]+"-final{epoch:02d}-{val_loss:.2f}",
        mode="min",)


    trainer = Trainer(
        max_epochs=epoch,
        accelerator="gpu",
        devices=2,  # Use 2 GPUs
        strategy="ddp",  # Use DDP (Distributed Data Parallel) DDPStrategy(find_unused_parameters=True)
        callbacks=[checkpoint_callback, lr_monitor],
        # precision='16',  # Enables mixed precision on Ampere+ GPUs
        logger=wandb_logger,
        profiler="simple", #enable_progress_bar=True,
        enable_model_summary=False,
        accumulate_grad_batches=8 , # Accumulate gradients over 8 smaller batches
    )
    # model = Lightning(config,SeparationNet)
    model = Lightning(config,SeparationNet)

    ###FOR REGULAR TRAINING #####

    print("started fitting, model on gpu: ")
    # trainer.fit(model = model.cuda(),ckpt_path=config["Best_epoch"])#,,ckpt_path=config["Best_epoch"]

    # ###FOR FINETUNING######
    model = model.load_from_checkpoint(config["Best_epoch"], config=config, SeparationNet=SeparationNet)
    trainer.fit(model=model.cuda())

    print("stopped fitting")
    trainer.save_checkpoint(config["save_dir"]+config["save_name"]+".ckpt")



if __name__ == "__main__":
    Train()

