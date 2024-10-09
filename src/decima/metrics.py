import torch
from torch import tensor, nn
from grelu.lightning.metrics import MSE
from torchmetrics import Metric


class DiseaseLfcMSE(Metric):
    
    def __init__(self, pairs, average: bool = True) -> None:
        super().__init__()
        self.mse = MSE(num_outputs = 1, average=False)
        self.disease = pairs[:, 0]
        self.healthy = pairs[:, 1]
        self.average = average

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        pred_lfcs = preds[:, self.disease, 0] - preds[:, self.healthy, 0] # B, T
        target_lfcs = target[:, self.disease, 0] - target[:, self.healthy, 0] # B, T
        self.mse.update(pred_lfcs, target_lfcs)
        
    def compute(self) -> torch.Tensor:
        output = self.mse.compute()
        if self.average:
            return output.mean()
        else:
            return output

    def reset(self) -> None:
        self.mse.reset()