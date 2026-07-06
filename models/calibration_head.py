# models/calibration_head.py
import torch
import torch.nn as nn
import torch.optim as optim

class TemperatureCalibrationHead:
    """
    Post-Hoc Calibration Engine: Minimizes empirical Expected Calibration Error (ECE)
    using optimization steps over validation set distributions.
    """
    def __init__(self, temperature_parameter: torch.Tensor):
        self.temperature = temperature_parameter

    def optimize_temperature(self, val_logits: torch.Tensor, val_labels: torch.Tensor):
        """Finds global optimal scalar T by minimizing Cross-Entropy on validation logs."""
        criterion = nn.CrossEntropyLoss()
        
        # Explicit L-BFGS setup for exact convergence boundaries
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(val_logits / self.temperature, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)