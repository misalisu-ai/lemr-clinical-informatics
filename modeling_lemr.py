# modeling_lemr.py
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from configuration_lemr import LEMRConfig
from models.fusion_gmu import GatedMultimodalUnit

class LEMRPreTrainedModel(PreTrainedModel):
    config_class = LEMRConfig
    base_model_prefix = "lemr"

class LEMRForClinicalPrediction(LEMRPreTrainedModel):
    def __init__(self, config: LEMRConfig):
        super().__init__(config)
        self.config = config
        self.fusion = GatedMultimodalUnit(config.vitals_dim, config.text_dim, config.latent_dim)
        self.classifier = nn.Linear(config.latent_dim, config.num_classes)
        
        # Non-trainable explicit parameter for post-hoc validation updates
        self.temperature = nn.Parameter(torch.tensor([config.temperature]), requires_grad=False)

    def forward(self, vitals_features: torch.Tensor, text_features: torch.Tensor, labels: torch.Tensor = None):
        fused_representation = self.fusion(vitals_features, text_features)
        logits = self.classifier(fused_representation)
        
        # Soften confidence predictions without altering class order
        calibrated_logits = logits / self.temperature
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(calibrated_logits, labels)

        return {
            "loss": loss, 
            "logits": calibrated_logits, 
            "uncalibrated_logits": logits
        }