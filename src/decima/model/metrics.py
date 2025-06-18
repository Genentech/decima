from enum import Enum
from collections import Counter
from typing import List
import torch
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
