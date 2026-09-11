"""Calibrate a fixed checkpoint using one cached validation forward pass."""

import numpy as np
import torch
from scipy.optimize import minimize_scalar
from scipy.special import expit

from graphormer.config.hparams import HyperparameterConfig
from graphormer.results import calculate_ece


class TemperatureOptimizer:
    def __init__(self, hparam_config: HyperparameterConfig):
        device = torch.device(hparam_config.torch_device)
        data_config = hparam_config.data_config()
        _, validation_loader = data_config.build()
        model = (hparam_config.model_config().build(node_feature_dim=data_config.num_node_features, edge_feature_dim=data_config.num_edge_features, output_dim=1).to(device).eval())
        labels, logits = [], []
        with torch.inference_mode():
            for batch in validation_loader:
                batch = batch.to(device)
                labels.append(batch.y.flatten().cpu())
                logits.append(model(batch).flatten().cpu())
        if not logits:
            raise ValueError("Temperature calibration requires a nonempty validation set")
        self.labels = torch.cat(labels).numpy()
        self.logits = torch.cat(logits).numpy()

    def get_predictions(self, temperature: float) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        return self.labels, expit(self.logits / temperature)

    def objective(self, temperature: float) -> float:
        return calculate_ece(*self.get_predictions(temperature))

    def optimize(self, max_iter: int = 50) -> float:
        result = minimize_scalar(self.objective, bounds=(0.1, 5.0), method="bounded", options={"maxiter": max_iter})
        print(f"Optimal Temperature: {result.x:.4f}, ECE: {result.fun:.4f}")
        return float(result.x)


def optimise_temperature(hparam_config: HyperparameterConfig, max_iter: int = 50) -> float:
    return TemperatureOptimizer(hparam_config).optimize(max_iter)
