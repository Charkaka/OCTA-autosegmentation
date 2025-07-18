from models.model_interface_abc import ModelInterface, Output
from models.base_model_abc import BaseModelABC
from typing import Any, Callable, Tuple
import torch
from utils.decorators import overrides
from utils.losses import get_loss_function_by_name
from utils.metrics import MetricsManager
from utils.visualizer import Visualizer
from torch.cuda.amp.grad_scaler import GradScaler
from monai.data import decollate_batch
from utils.enums import Phase

class LambdaModel(BaseModelABC):

    def __init__(self, model_name: str, phase: Phase, MODEL_DICT: dict, inference: str="model", **kwargs) -> None:
        super().__init__(optimizer_mapping={"optimizer": ["model"]})
        self.model = MODEL_DICT[model_name](**kwargs)
        # Set netG_A if inference mode is netG_A
        if inference == "netG_A":
            self.netG_A = self.model
        # Optionally, always set netG_A for compatibility
        else:
            self.netG_A = self.model

    overrides(BaseModelABC)
    def initialize_model_and_optimizer(self, init_mini_batch: dict, init_weights: Callable, config: dict[str, dict], args, scaler: GradScaler, phase:Phase=Phase.TRAIN) -> None:
        if phase!=Phase.TEST:
            self.loss_name = config.get(Phase.TRAIN, dict()).get("loss", "")
            self.loss_function = get_loss_function_by_name(self.loss_name, config)
        if phase==Phase.TRAIN and config[Phase.TRAIN].get("AT", False):
            self.at = get_loss_function_by_name("AtLoss", config, scaler, self.loss_function)
        super().initialize_model_and_optimizer(init_mini_batch,init_weights, config, args, scaler, phase)

    overrides(BaseModelABC)
    def inference(self,
            mini_batch: dict[str, Any],
            post_transformations: dict[str, Callable],
            device: torch.device = "cpu",
            phase: Phase = Phase.TEST
        ) -> Tuple[Output, dict[str, torch.Tensor]]:
        inputs =  mini_batch["image"].to(device, non_blocking=True)
        if phase!=Phase.TEST:
            labels = mini_batch["label"].to(device, non_blocking=True)
        if phase==Phase.TRAIN and hasattr(self, "at"):
            inputs, labels = self.at(self.model, inputs, mini_batch["background"].to(device, non_blocking=True), labels)
            mini_batch["image"] = inputs.detach()
        pred = self.model(inputs).squeeze(-1)
        outputs: Output = { "prediction": [post_transformations["prediction"](i) for i in decollate_batch(pred[0:1])] }
        if phase != Phase.TEST:
            outputs["label"] = [post_transformations["label"](i) for i in decollate_batch(labels[0:1])]
            losses = { self.loss_name: self.loss_function(y_pred=pred, y=labels) }
        else:
            losses = None
        return outputs, losses
    
    overrides(BaseModelABC)
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.model(input)
    
    overrides(ModelInterface)
    def compute_metric(self, outputs: Output, metrics: MetricsManager) -> None:
        metrics(outputs["prediction"], outputs["label"])
    
    overrides(ModelInterface)
    def plot_sample(self,
            visualizer: Visualizer,
            mini_batch: dict[str, Any],
            outputs: Output,
            *,
            suffix: str = ""
        ) -> str:
        return visualizer.plot_sample(
            mini_batch["image"][0],
            outputs["prediction"][0],
            outputs["label"][0],
            suffix=suffix,
        )
