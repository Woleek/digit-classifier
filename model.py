import os
import torch
import pytorch_lightning as pl

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics.functional import accuracy
from torch.optim import Adam


class MNIST_Net(pl.LightningModule):
    
    def __init__(self):
        super(MNIST_Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=128, 
                kernel_size=(3,3),
                stride=1),
            nn.MaxPool2d(kernel_size=(3,3)),
            nn.Conv2d(
                in_channels=128, 
                out_channels=256, 
                kernel_size=(3,3),
                stride=2),
            nn.MaxPool2d(kernel_size=(3,3)),
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(in_features=256,
                      out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128,
                      out_features=10),
        )
        self.save_hyperparameters()
    
    def prepare_data(self):
        import pandas as pd
        train_df = pd.read_csv('data\\train.csv')
        y_train_df = train_df['label']
        X_train_df = train_df.drop('label', axis=1)/255
        X_test_df = pd.read_csv('data\\test.csv')
        
        y_train = torch.Tensor(y_train_df.values)
        X_train = torch.Tensor(X_train_df.values).reshape(-1,1,28,28)
        X_test = torch.Tensor(X_test_df.values).reshape(-1,1,28,28)
        
        dataset = TensorDataset(X_train, y_train)
        self.train_set, self.val_set = random_split(dataset, [0.9, 0.1])
        self.test_set = TensorDataset(X_test)
        
    def train_dataloader(self):
        train_loader = DataLoader(
            dataset=self.train_set, 
            shuffle=True,
            batch_size=100,
            num_workers=4,
        )
        return train_loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.val_set,
            shuffle=False,
            batch_size=100,
            num_workers=4,
        )
        return val_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(
            dataset=self.test_set,
            shuffle=False,
            batch_size=100,
            num_workers=4,
        )
        return test_loader
    
    def forward(self, x):
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y = y.type(torch.LongTensor).to(self.device)
        y_hat = self(X)
        
        train_acc = accuracy(y_hat, y, num_classes=10)
        loss = F.cross_entropy(y_hat, y)
        
        self.log_dict({'train_loss': loss, 'train_acc': train_acc}, prog_bar=True)
        return {"loss": loss, "acc": train_acc}
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y = y.type(torch.LongTensor).to(self.device)
        y_hat = self(X)
        
        val_acc = accuracy(y_hat, y, num_classes=10)
        loss = F.cross_entropy(y_hat, y)
        
        self.log_dict({'val_loss': loss, 'val_acc': val_acc}, on_epoch=True)
        return {"loss": loss, "acc": val_acc}
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return {'optimizer': optimizer}