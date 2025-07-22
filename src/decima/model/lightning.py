"""
The LightningModel class.
"""

import json
from datetime import datetime
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
from einops import rearrange
from grelu.lightning.metrics import MSE, PearsonCorrCoef
from grelu.utils import make_list
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchmetrics import MetricCollection
import safetensors

from .decima_model import DecimaModel
from .loss import TaskWisePoissonMultinomialLoss
from .metrics import DiseaseLfcMSE, WarningCounter, GenePearsonCorrCoef


default_train_params = {
    "lr": 4e-5,
    "batch_size": 4,
    "num_workers": 1,
    "devices": 0,
    "logger": "csv",
    "save_dir": ".",
    "max_epochs": 15,
    "accumulate_grad_batches": 1,
    "total_weight": 1e-4,
    "disease_weight": 1e-2,
    "clip": 0.0,
    "save_top_k": 1,
    "pin_memory": True,
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

    def __init__(self, model_params: dict, train_params: dict = {}, data_params: dict = {}, name: str = "") -> None:
        super().__init__()

        self.name = name
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
        self.loss = TaskWisePoissonMultinomialLoss(total_weight=self.train_params["total_weight"], debug=True)
        self.val_losses = []
        self.test_losses = []

        # Set up activation function
        self.activation = torch.exp

        # Inititalize metrics
        _metrics = {
            "mse": MSE(num_outputs=self.model.head.n_tasks, average=False),
            "task_pearson": PearsonCorrCoef(num_outputs=self.model.head.n_tasks, average=False),
            "gene_pearson": GenePearsonCorrCoef(average=False),
        }
        if "pairs" in self.train_params:
            _metrics["disease_lfc_mse"] = DiseaseLfcMSE(pairs=self.train_params["pairs"], average=False)
        metrics = MetricCollection(_metrics)
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
        self.warning_counter = WarningCounter()

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
            logger = CSVLogger(name=self.train_params["name"], save_dir=self.train_params["save_dir"])
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
            pin_memory=self.train_params.get("pin_memory", False),
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
            pin_memory=self.train_params.get("pin_memory", False),
        )

    def make_predict_loader(
        self,
        dataset: Callable,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
        **kwargs,
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
            **kwargs,
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
            accelerator="gpu",
            devices=make_list(self.train_params["devices"]),
            logger=logger,
            callbacks=[ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=self.train_params["save_top_k"])],
            default_root_dir=self.train_params["save_dir"],
            accumulate_grad_batches=self.train_params["accumulate_grad_batches"],
            gradient_clip_val=self.train_params["clip"],
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
        if "tasks" not in self.data_params:
            self.data_params["tasks"] = train_dataset.tasks.reset_index(names="name").to_dict(orient="list")

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

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Union[dict, Tensor]:
        """
        Predict for a single batch of sequences or variants

        Args:
            batch: Batch of sequences or variants
            batch_idx: Index of the batch
            dataloader_idx: Index of the dataloader

        Returns:
            Dictionary containing predictions and warnings or a tensor of predictions
        """
        if isinstance(batch, dict):
            seq = batch["seq"]
            if "pred_expr" in batch:
                pred_expr = batch["pred_expr"][self.name]
                pred_expr = self.transform(pred_expr.unsqueeze(-1))
                expression = torch.zeros_like(pred_expr)
                precomputed = ~pred_expr.isnan()
                expression[precomputed] = pred_expr[precomputed]
                expression[~precomputed] = self(seq[~precomputed.all(axis=(1, 2))]).view(-1)
            else:
                expression = self(seq)
            return {"expression": expression, "warnings": batch["warning"]}
        else:
            return self(batch)

    def predict_on_dataset(
        self,
        dataset: Callable,
        devices: Optional[int] = None,
        num_workers: int = 1,
        batch_size: int = 6,
        augment_aggfunc: Union[str, Callable] = "mean",
        compare_func: Optional[Union[str, Callable]] = None,
        float_precision: str = "32",
    ):
        """
        Predict for a dataset of sequences or variants

        Args:
            dataset: Dataset object that yields one-hot encoded sequences

            devices: Number of devices to use,
                e.g. machine has 4 gpu's but only want to use 2 for predictions

            num_workers: Number of workers for data loader

            batch_size: Batch size for data loader

        Returns:
            Model predictions as a numpy array or dataframe
        """
        torch.set_float32_matmul_precision("medium")

        accelerator = "auto"
        if devices is None:
            devices = "auto"  # use all devices
            accelerator = "gpu" if torch.cuda.is_available() else "auto"

        if accelerator == "auto":
            trainer = pl.Trainer(accelerator=accelerator, logger=False, precision=float_precision)
        else:
            trainer = pl.Trainer(accelerator=accelerator, devices=devices, logger=False, precision=float_precision)

        # Make dataloader
        dataloader = self.make_predict_loader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else default_collate,
        )

        # Predict
        results = trainer.predict(self, dataloader)
        if isinstance(results[0], dict):
            expression = torch.concat([r["expression"] for r in results]).squeeze(-1)

            for r in results:
                self.warning_counter.update(r["warnings"])
        else:
            expression = torch.concat(results).squeeze(-1)

        # Reshape predictions
        expression = rearrange(
            expression,
            "(b n a) t -> b n a t",
            n=dataset.n_augmented,
            a=dataset.n_alleles,
        )

        # Convert predictions to numpy array
        expression = expression.detach().cpu().float().numpy()

        if dataset.n_alleles == 2:
            expression = expression[:, :, 1, :] - expression[:, :, 0, :]  # BNT
        else:
            expression = expression.squeeze(2)  # B N T

        expression = np.mean(expression, axis=-2)  # B T

        return {"expression": expression, "warnings": self.warning_counter.compute()}

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
            return [i for i in range(self.model_params["n_tasks"]) if i not in make_list(tasks)]

    @classmethod
    def load_safetensor(cls, path: str, device: str = "cpu"):
        with safetensors.safe_open(path, framework="pt") as f:
            metadata = f.metadata()
            model = cls(
                name=metadata["name"],
                model_params=json.loads(metadata["model_params"]),
                data_params=json.loads(metadata["data_params"]),
            )
        state_dict = safetensors.torch.load_file(path)
        model.model.load_state_dict(state_dict)
        return model.to(device)


class EnsembleLightningModel(LightningModel):
    def __init__(self, models: List[LightningModel]):
        super().__init__(
            model_params=models[0].model_params,
            train_params=models[0].train_params,
            data_params=models[0].data_params,
        )
        self.models = nn.ModuleList(models)
        self.reset_transform()
        self.name = "ensemble"

    def forward(self, x: Tensor) -> Tensor:
        return torch.concat([model(x) for model in self.models], dim=0)

    def training_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        raise NotImplementedError("Ensemble training is not implemented.")

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        raise NotImplementedError("Ensemble validation is not implemented.")

    def test_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        raise NotImplementedError("Ensemble test is not implemented.")

    def predict_on_dataset(
        self,
        dataset: Callable,
        devices: Optional[int] = None,
        num_workers: int = 1,
        batch_size: int = 6,
        augment_aggfunc: Union[str, Callable] = "mean",
        compare_func: Optional[Union[str, Callable]] = None,
        float_precision: str = "32",
    ):
        preds = super().predict_on_dataset(
            dataset=dataset,
            devices=devices,
            num_workers=num_workers,
            batch_size=batch_size,
            augment_aggfunc=augment_aggfunc,
            compare_func=compare_func,
            float_precision=float_precision,
        )
        expression = rearrange(
            preds["expression"],
            "(e b) t -> e b t",
            e=len(self.models),
        )
        return {
            "expression": expression.mean(axis=0),
            "warnings": preds["warnings"],
            "ensemble_preds": expression,
        }

    @classmethod
    def load_from_checkpoints(cls, checkpoints: List[str]):
        models = []
        for checkpoint in checkpoints:
            models.append(LightningModel.load_from_checkpoint(checkpoint))
        return cls(models)

    def add_transform(self, prediction_transform: Callable) -> None:
        for model in self.models:
            model.add_transform(prediction_transform)

    def reset_transform(self) -> None:
        if hasattr(self, "models"):
            for model in self.models:
                model.reset_transform()

    def transform(self, x: Tensor) -> Tensor:
        return self.models[0].transform(x)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Union[dict, Tensor]:
        """
        Predict for a single batch of sequences or variants

        Args:
            batch: Batch of sequences or variants
            batch_idx: Index of the batch
            dataloader_idx: Index of the dataloader

        Returns:
            Dictionary containing predictions and warnings or a tensor of predictions
        """
        if isinstance(batch, dict):
            expression = torch.concat(
                [model.predict_step(batch, batch_idx, dataloader_idx)["expression"] for model in self.models]
            )
            return {"expression": expression, "warnings": batch["warning"]}
        else:
            return self(batch)
