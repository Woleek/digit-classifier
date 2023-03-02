from model import MNIST_Net

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

if __name__ == '__main__':
    i = 0
    while os.path.exists(f".\\lightning_logs\\version_{i}"):
        i += 1
    
    model = MNIST_Net()
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=20,
        devices='auto',
        fast_dev_run=False,
        logger = TensorBoardLogger(
            save_dir='.'
        ),
        callbacks=[ModelCheckpoint(
            dirpath=f".\\lightning_logs\\version_{i}",
            filename=f'best_checkpoint',
            save_top_k=1)],
        log_every_n_steps=50,
        auto_lr_find=True,
    )
    trainer.fit(model)
    
    torch.save(model, f".\\lightning_logs\\version_{i}\\model.pt")
    
    os.system('PAUSE')