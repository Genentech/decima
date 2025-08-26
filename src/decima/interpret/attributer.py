import warnings
import torch
from captum.attr import InputXGradient, Saliency, IntegratedGradients
from grelu.transforms.prediction_transforms import Aggregate, Specificity

from decima.hub import load_decima_model


def get_attribution_method(method: str):
    """Get attribution method from string.

    Args:
        method: Method to use for attribution analysis

    Returns:
        Attribution: Attribution analysis results for the gene and tasks
    """
    if method == "saliency":
        return Saliency
    elif method == "inputxgradient":
        return InputXGradient
    elif method == "integratedgradients":
        return IntegratedGradients
    return method


class DecimaAttributer:
    def __init__(self, model, tasks, off_tasks=None, method: str = "inputxgradient", transform="specificity"):
        self.model = model
        self.method = method
        self.transform = transform

        if transform == "specificity":
            self.model.add_transform(
                Specificity(
                    on_tasks=tasks,
                    off_tasks=off_tasks,
                    model=model,
                    compare_func="subtract",
                )
            )
            if off_tasks is None:
                warnings.warn("`off_tasks` is not provided. Using all other tasks as off_tasks.")

        elif transform == "aggregate":
            if off_tasks is not None:
                raise ValueError(
                    "`off_tasks` is not allowed with `aggregate` transform. Please use `specificity` instead."
                )
            self.model.add_transform(Aggregate(tasks=tasks, task_aggfunc="mean", model=model))

        self.model.eval()

        attribution_method = get_attribution_method(method)
        self._attributer = attribution_method(self.model)

    def attribute(self, inputs, **kwargs):
        if self.method == "saliency":
            kwargs = {**kwargs, "abs": False}

        with torch.no_grad():
            inputs.requires_grad = True
            return self._attributer.attribute(inputs, **kwargs)[:, :4]

    @classmethod
    def load_decima_attributer(
        cls,
        model_name,
        tasks,
        off_tasks=None,
        method: str = "inputxgradient",
        transform="specificity",
        device="cpu",
    ):
        return cls(load_decima_model(model_name, device=device), tasks, off_tasks, method, transform)
