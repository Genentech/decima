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
    """
    DecimaAttributer class for attribution analysis.

    Args:
        model: Model to attribute.
        tasks: Tasks to attribute.
        off_tasks: Off tasks to attribute.
        method: Method to use for attribution analysis available options: "saliency", "inputxgradient", "integratedgradients".
        transform: Transform to use for attribution analysis.

    Examples:
        >>> attributer = DecimaAttributer(
        ...     model,
        ...     tasks,
        ...     off_tasks,
        ...     method,
        ...     transform,
        ... )
        >>> attributer.attribute(
        ...     inputs
        ... )
    """

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
        """Attribute inputs.

        Args:
            inputs: Inputs to attribute.
            **kwargs: Additional arguments to pass to the attribution method.

        Returns:
            torch.Tensor: Attribution analysis results for the gene and tasks
        """
        inputs = inputs.to(self.model.device)

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
        """Load DecimaAttributer.

        Args:
            model_name: Model name to load.
            tasks: Tasks to attribute.
            off_tasks: Off tasks to attribute.
            method: Method to use for attribution analysis available options: "saliency", "inputxgradient", "integratedgradients".
            transform: Transform to use for attribution analysis.
            device: Device to use for attribution analysis.
        """
        return cls(load_decima_model(model_name, device=device), tasks, off_tasks, method, transform)
