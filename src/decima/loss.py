import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TaskWisePoissonMultinomialLoss(nn.Module):
    def __init__(
        self,
        total_weight: float = 1,
        eps: float = 1e-7,
        debug=False,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.total_weight = total_weight
        self.debug = debug

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = torch.exp(input).squeeze(-1)  # B, T
        target = target.squeeze(-1)  # B, T

        total_target = target.sum(axis=-1)  # B,
        total_input = input.sum(axis=-1)  # B,

        # total count poisson loss, mean across targets
        poisson_term = F.poisson_nll_loss(total_input, total_target, log_input=False, reduction="mean")  # B
        poisson_term = self.total_weight * poisson_term  # B,

        # Get multinomial probabilities
        p_input = input / total_input.unsqueeze(1)  # B, T
        log_p_input = torch.log(p_input)  # B, T

        # multinomial loss
        multinomial_dot = -torch.multiply(target, log_p_input)  # B x T
        multinomial_term = multinomial_dot.mean()

        # Combine
        loss = multinomial_term + poisson_term
        if self.debug:
            print(f"Multinomial: {multinomial_term}, Poisson: {poisson_term}")
        return loss
