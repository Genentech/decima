from enum import Enum
from collections import Counter
from typing import List
import torch
import torchmetrics
from torchmetrics import Metric
from grelu.lightning.metrics import MSE


class DiseaseLfcMSE(Metric):
    def __init__(self, pairs, average: bool = True) -> None:
        super().__init__()
        self.mse = MSE(num_outputs=1, average=False)
        self.disease = pairs[:, 0]
        self.healthy = pairs[:, 1]
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pred_lfcs = preds[:, self.disease, 0] - preds[:, self.healthy, 0]  # B, T
        target_lfcs = target[:, self.disease, 0] - target[:, self.healthy, 0]  # B, T
        self.mse.update(pred_lfcs, target_lfcs)

    def compute(self) -> torch.Tensor:
        output = self.mse.compute()
        if self.average:
            return output.mean()
        else:
            return output

    def reset(self) -> None:
        self.mse.reset()


class WarningType(Enum):
    UNKNOWN = "unknown"
    ALLELE_MISMATCH_WITH_REFERENCE_GENOME = "allele_mismatch_with_reference_genome"


class WarningCounter(Metric):
    """
    A TorchMetric to count occurrences of different warning types,
    including a dedicated category for 'unknown' warnings.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False

    def __init__(self, warning_types: List[WarningType] = None, **kwargs):
        super().__init__(**kwargs)
        self.warning_types = warning_types or list(WarningType)

        self._idx = {wt: i for i, wt in enumerate(self.warning_types)}
        self.num_warning_types = len(self.warning_types)

        self.add_state("counts", default=torch.zeros(len(self.warning_types), dtype=torch.long), dist_reduce_fx="sum")

    def update(self, warnings: List[WarningType]):
        """
        Update the internal state with new warnings.

        Args:
            warnings: A list of warning strings from a batch.
        """
        warning_indices = torch.tensor(
            [
                self._idx[warning_type] if warning_type in self._idx else self._idx[WarningType.UNKNOWN]
                for warning_type in warnings
            ],
            dtype=torch.long,
            device=self.counts.device,
        )

        if warning_indices.numel() > 0:
            self.counts += torch.bincount(warning_indices, minlength=self.num_warning_types).to(self.counts.device)

    def compute(self) -> Counter:
        """
        Compute the final counts of all warning types.
        """
        return Counter({wt.value: self.counts[i] for i, wt in enumerate(self.warning_types)})


class GenePearsonCorrCoef(Metric):
    """
    Metric class to calculate the Pearson correlation coefficient for each gene.

    Args:
        average: If true, return the average metric across genes.
            Otherwise, return a separate value for each gene

    As input to forward and update the metric accepts the following input:
        preds: Predictions of shape (N, n_tasks, L)
        target: Ground truth labels of shape (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the Pearson coefficient.
    """

    def __init__(self, average = True) -> None:
        super().__init__()
        self.corrs = []
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        #preds: B, T, L
        preds = preds.flatten(start_dim=1, end_dim=2) # B, T
        target = target.flatten(start_dim=1, end_dim=2) # B, T
        n = preds.shape[0]
        corrs = [torch.corrcoef(torch.vstack([preds[i],target[i]]))[0,1] for i in range(n)]
        self.corrs.extend(corrs)

    def compute(self) -> torch.Tensor:
        corrs = torch.tensor(self.corrs)
        if self.average:
            corrs = corrs.mean()
        return corrs

    def reset(self) -> None:
        self.corrs = []


class TaskPearsonCorrCoef(Metric):
    """
    Metric class to calculate the Pearson correlation coefficient for each task.

    Args:
        num_outputs: Number of tasks
        average: If true, return the average metric across tasks.
            Otherwise, return a separate value for each task

    As input to forward and update the metric accepts the following input:
        preds: Predictions of shape (N, n_tasks, L)
        target: Ground truth labels of shape (N, n_tasks, L)

    As output of forward and compute the metric returns the following output:
        output: A tensor with the Pearson coefficient.
    """

    def __init__(self, num_outputs: int = 1, average: bool = True) -> None:
        super().__init__()
        self.pearson = torchmetrics.PearsonCorrCoef(num_outputs=num_outputs)
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        preds = (
            preds.swapaxes(1, 2).flatten(start_dim=0, end_dim=1).to(torch.float32)
        )  # Nx L, n_tasks
        target = (
            target.swapaxes(1, 2).flatten(start_dim=0, end_dim=1).to(torch.float32)
        )  # Nx L, n_tasks
        var_x = self.pearson.var_x
        var_y = self.pearson.var_y
        corr_xy = self.pearson.corr_xy
        n_total = self.pearson.n_total
        var_x = var_x / (n_total - 1)
        var_y = var_y / (n_total - 1)
        corr_xy = corr_xy / (n_total - 1)

        corrcoef = ((corr_xy / (var_x * var_y).sqrt()).squeeze().to(torch.float32))
        corrcoef = torch.clamp(corrcoef, -1.0, 1.0)
        return corrcoef.squeeze()

    def compute(self) -> torch.Tensor:
        output = self.pearson.compute()
        if self.average:
            return output.mean()
        else:
            return output

    def reset(self) -> None:
        self.pearson.reset()
