"""
The LightningModel class.
"""

import warnings
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from einops import rearrange
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection

from grelu.lightning.metrics import MSE, PearsonCorrCoef
from grelu.sequence.format import strings_to_one_hot
from grelu.utils import get_aggfunc, get_compare_func, make_list

import os, sys
sys.path.append(os.path.dirname(__file__))
sys.path.insert(0, '/code/decima/src/decima')

from decima_model import DecimaModel
from loss import TaskWisePoissonMultinomialLoss
from read_hdf5 import VariantDataset
from metrics import DiseaseLfcMSE


default_train_params = {
    "lr": 4e-5,
    "batch_size": 4,
    "num_workers": 1,
    "devices": 0,
    "logger": "csv",
    "save_dir": ".",
    "max_epochs": 1,
    "accumulate_grad_batches":1,
    "total_weight": 1e-4,
    "disease_weight":1e-2,
}


class LightningModel(pl.LightningModule):
    """
    Wrapper for predictive sequence models

    Args:
        model_params: Dictionary of parameters specifying model architecture
        train_params: Dictionary specifying training parameters
        data_params: Dictionary specifying parameters of the training data.
            This is empty by default and will be filled at the time of
            training.
    """

    def __init__(
        self, model_params: dict, train_params: dict = {}, data_params: dict = {}
    ) -> None:
        super().__init__()

        self.save_hyperparameters(ignore=["model"])

        # Add default training parameters
        for key in default_train_params.keys():
            if key not in train_params:
                train_params[key] = default_train_params[key]

        # Save params
        self.model_params = model_params
        self.train_params = train_params
        self.data_params = data_params

        # Build model
        self.model = DecimaModel(**{k: v for k, v in self.model_params.items()})

        # Set up loss function
        self.loss = TaskWisePoissonMultinomialLoss(
                total_weight=self.train_params["total_weight"],
                debug=True
                )
        self.val_losses = []
        self.test_losses = []

        # Set up activation function
        self.activation = torch.exp

        # Inititalize metrics
        metrics = MetricCollection(
            {
                "mse": MSE(num_outputs=self.model.head.n_tasks, average=False),
                "pearson": PearsonCorrCoef(
                    num_outputs=self.model.head.n_tasks, average=False
                ),
                "disease_lfc_mse": DiseaseLfcMSE(
                    pairs=self.train_params["pairs"], average=False
                )
            }
        )
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Initialize prediction transform
        self.reset_transform()
        

    def format_input(self, x: Union[Tuple[Tensor, Tensor], Tensor]) -> Tensor:
        """
        Extract the one-hot encoded sequence from the input
        """
        # if x is a tuple of sequence, label, return the sequence
        if isinstance(x, Tensor):
            if x.ndim == 3:
                return x
            else:
                return x.unsqueeze(0)
        elif isinstance(x, Tuple):
            return x[0]
        else:
            raise Exception("Cannot perform forward pass on the given input format.")

    def forward(
        self,
        x: Union[Tuple[Tensor, Tensor], Tensor, str, List[str]],
        logits: bool = False,
    ) -> Tensor:
        """
        Forward pass
        """
        # Format the input as a one-hot encoded tensor
        x = self.format_input(x)

        # Run the model
        x = self.model(x)

        # forward() produces prediction (e.g. post-activation)
        # unless logits=True, which is used in loss functions
        if not logits:
            x = self.activation(x)

        # Apply transform
        x = self.transform(x)
        return x

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        logits = self.forward(x, logits=True)
        loss = self.loss(logits, y)
        self.log("train_loss", loss, logger=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        x, y = batch
        logits = self.forward(x, logits=True)
        loss = self.loss(logits, y)
        y_hat = self.activation(logits)
        self.log("val_loss", loss, logger=True, on_step=False, on_epoch=True)
        self.val_metrics.update(y_hat, y)
        self.val_losses.append(loss)
        return loss

    def on_validation_epoch_end(self):
        """
        Calculate metrics for entire validation set
        """
        # Compute metrics
        val_metrics = self.val_metrics.compute()
        mean_val_metrics = {k: v.mean() for k, v in val_metrics.items()}
        # Compute loss
        losses = torch.stack(self.val_losses)
        mean_losses = torch.mean(losses)
        # Log
        self.log_dict(mean_val_metrics)
        self.log("val_loss", mean_losses)

        self.val_metrics.reset()
        self.val_losses = []

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        """
        Calculate metrics after a single test step
        """
        x, y = batch
        logits = self.forward(x, logits=True)
        loss = self.loss(logits, y)
        y_hat = self.activation(logits)
        self.log("test_loss", loss, logger=True, on_step=False, on_epoch=True)
        self.test_metrics.update(y_hat, y)
        self.test_losses.append(loss)
        return loss

    def on_test_epoch_end(self) -> None:
        """
        Calculate metrics for entire test set
        """
        self.computed_test_metrics = self.test_metrics.compute()
        self.log_dict({k: v.mean() for k, v in self.computed_test_metrics.items()})
        losses = torch.stack(self.test_losses)
        self.log("test_loss", torch.mean(losses))
        self.test_metrics.reset()
        self.test_losses = []

    def configure_optimizers(self) -> None:
        """
        Configure oprimizer for training
        """
        return optim.Adam(self.parameters(), lr=self.train_params["lr"])

    def count_params(self) -> int:
        """
        Number of gradient enabled parameters in the model
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def parse_logger(self) -> str:
        """
        Parses the name of the logger supplied in train_params.
        """
        if "name" not in self.train_params:
            self.train_params["name"] = datetime.now().strftime("%Y_%d_%m_%H_%M")
        if self.train_params["logger"] == "wandb":
            logger = WandbLogger(
                name=self.train_params["name"],
                log_model=True,
                save_dir=self.train_params["save_dir"],
            )
        elif self.train_params["logger"] == "csv":
            logger = CSVLogger(
                name=self.train_params["name"], save_dir=self.train_params["save_dir"]
            )
        else:
            raise NotImplementedError
        return logger

    def add_transform(self, prediction_transform: Callable) -> None:
        """
        Add a prediction transform
        """
        if prediction_transform is not None:
            self.transform = prediction_transform

    def reset_transform(self) -> None:
        """
        Remove a prediction transform
        """
        self.transform = nn.Identity()

    def make_train_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Callable:
        """
        Make dataloader for training
        """
        return DataLoader(
            dataset,
            batch_size=batch_size or self.train_params["batch_size"],
            shuffle=True,
            num_workers=num_workers or self.train_params["num_workers"],
        )

    def make_test_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Callable:
        """
        Make dataloader for validation and testing
        """
        return DataLoader(
            dataset,
            batch_size=batch_size or self.train_params["batch_size"],
            shuffle=False,
            num_workers=num_workers or self.train_params["num_workers"],
        )

    def make_predict_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> Callable:
        """
        Make dataloader for prediction
        """
        dataset.predict = True
        return DataLoader(
            dataset,
            batch_size=batch_size or self.train_params["batch_size"],
            shuffle=False,
            num_workers=num_workers or self.train_params["num_workers"],
        )

    def train_on_dataset(
        self,
        train_dataset: Callable,
        val_dataset: Callable,
        checkpoint_path: Optional[str] = None,
    ):
        """
        Train model and optionally log metrics to wandb.

        Args:
            train_dataset (Dataset): Dataset object that yields training examples
            val_dataset (Dataset) : Dataset object that yields training examples
            checkpoint_path (str): Path to model checkpoint from which to resume training.
                The optimizer will be set to its checkpointed state.

        Returns:
            PyTorch Lightning Trainer
        """
        torch.set_float32_matmul_precision("medium")
        
        # Set up logging
        logger = self.parse_logger()

        # Set up trainer
        trainer = pl.Trainer(
            max_epochs=self.train_params["max_epochs"],
            accelerator='gpu',
            devices=make_list(self.train_params["devices"]),
            logger=logger,
            callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_last=True)],
            default_root_dir=self.train_params["save_dir"],
            accumulate_grad_batches=self.train_params["accumulate_grad_batches"],
            precision="16-mixed",
        )

        # Make dataloaders
        train_dataloader = self.make_train_loader(train_dataset)
        val_dataloader = self.make_test_loader(val_dataset)

        if checkpoint_path is None:
            # First validation pass
            trainer.validate(model=self, dataloaders=val_dataloader)
            self.val_metrics.reset()

        # Add data parameters
        self.data_params["tasks"] = train_dataset.tasks.reset_index(
            names="name"
        ).to_dict(orient="list")

        for attr, value in self._get_dataset_attrs(train_dataset):
            self.data_params["train_" + attr] = value

        for attr, value in self._get_dataset_attrs(val_dataset):
            self.data_params["val_" + attr] = value

        # Training
        trainer.fit(
            model=self,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=checkpoint_path,
        )
        return trainer

    def _get_dataset_attrs(self, dataset: Callable) -> None:
        """
        Read data parameters from a dataset object
        """
        for attr in dir(dataset):
            if not attr.startswith("_") and not attr.isupper():
                value = getattr(dataset, attr)
                if (
                    (isinstance(value, str))
                    or (isinstance(value, int))
                    or (isinstance(value, float))
                    or (value is None)
                ):
                    yield attr, value

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        checkpoint["hyper_parameters"]["data_params"] = self.data_params

    def predict_on_dataset(
        self,
        dataset: Callable,
        devices: int = 0,
        num_workers: int = 1,
        batch_size: int = 6,
        augment_aggfunc: Union[str, Callable] = "mean",
        compare_func: Optional[Union[str, Callable]] = None,
    ):
        """
        Predict for a dataset of sequences or variants

        Args:
            dataset: Dataset object that yields one-hot encoded sequences
            devices: Device IDs to use
            num_workers: Number of workers for data loader
            batch_size: Batch size for data loader

        Returns:
            Model predictions as a numpy array or dataframe
        """
        torch.set_float32_matmul_precision("medium")
        dataloader = self.make_predict_loader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
        )
        trainer = pl.Trainer(accelerator="gpu", devices=make_list(devices), logger=None)

        # Predict
        preds = torch.concat(trainer.predict(self, dataloader)).squeeze(-1)

        # Reshape predictions
        preds = rearrange(
            preds,
            "(b n a) t -> b n a t",
            n=dataset.n_augmented,
            a=dataset.n_alleles,
        )

        # Convert predictions to numpy array
        preds = preds.detach().cpu().numpy()

        if dataset.n_alleles==2:
            preds = preds[:, :, 1, :] - preds[:, :, 0, :] # BNT
        else:
            preds = preds.squeeze(2)  # B N T

        preds = np.mean(preds, axis=1)  # B T
        return preds

    def get_task_idxs(
        self,
        tasks: Union[int, str, List[int], List[str]],
        key: str = "name",
        invert: bool = False,
    ) -> Union[int, List[int]]:
        """
        Given a task name or metadata entry, get the task index
        If integers are provided, return them unchanged

        Args:
            tasks: A string corresponding to a task name or metadata entry,
                or an integer indicating the index of a task, or a list of strings/integers
            key: key to model.data_params["tasks"] in which the relevant task data is
                stored. "name" will be used by default.
            invert: Get indices for all tasks except those listed in tasks

        Returns:
            The index or indices of the corresponding task(s) in the model's
            output.
        """
        # If a string is provided, extract the index
        if isinstance(tasks, str):
            return self.data_params["tasks"][key].index(tasks)
        # If an integer is provided, return it as the index
        elif isinstance(tasks, int):
            return tasks
        # If a list is provided, return teh index for each element
        elif isinstance(tasks, list):
            return [self.get_task_idxs(task) for task in tasks]
        else:
            raise TypeError("Input must be a list, string or integer")
        if invert:
            return [
                i
                for i in range(self.model_params["n_tasks"])
                if i not in make_list(tasks)
            ]
