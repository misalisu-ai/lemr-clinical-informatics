# models/fusion_gmu.py
import torch
import torch.nn as nn

class GatedMultimodalUnit(nn.Module):
    """
    Adaptive Convex Gated Fusion Layer: Weights features within the convex 
    hull based on modality completeness distributions.
    """
    def __init__(self, vitals_dim: int, text_dim: int, latent_dim: int):
        super().__init__()
        self.proj_vitals = nn.Linear(vitals_dim, latent_dim)
        self.proj_text = nn.Linear(text_dim, latent_dim)
        
        # Dual gating neural weight parameter map
        self.gate = nn.Linear(vitals_dim + text_dim, latent_dim)
        self.activation = nn.Tanh()

    def forward(self, x_vitals: torch.Tensor, x_text: torch.Tensor) -> torch.Tensor:
        h_vitals = self.activation(self.proj_vitals(x_vitals))
        h_text = self.activation(self.proj_text(x_text))

        # Joint latent combination calculation
        joint_context = torch.cat([x_vitals, x_text], dim=-1)
        alpha = torch.sigmoid(self.gate(joint_context))

        # Element-wise dynamic mathematical blend
        z = alpha * h_vitals + (1.0 - alpha) * h_text
        return z