# verify_pipeline.py
import torch
import torch.nn as nn
from configuration_lemr import LEMRConfig
from modeling_lemr import LEMRForClinicalPrediction
from data.noise_engine import ClinicalNoiseEngine
from models.calibration_head import TemperatureCalibrationHead

def parameter_update_hook(parameter):
    """PyTorch hook for in-place parameter updates, instantly freeing VRAM allocations."""
    if parameter.grad is not None:
        with torch.no_grad():
            parameter.data -= 0.001 * parameter.grad.data
        parameter.grad = None

if __name__ == "__main__":
    print("Initializing LEMR Architectural Testing Loop...")
    config = LEMRConfig(vitals_dim=24, text_dim=768, latent_dim=128)
    model = LEMRForClinicalPrediction(config)
    
    # 1. Register in-place gradient hook updates
    for param in model.parameters():
        if param.requires_grad:
            param.register_post_accumulate_grad_hook(parameter_update_hook)

    # 2. Simulate dirty, missing clinical tensors
    noise_engine = ClinicalNoiseEngine()
    clean_vitals = torch.randn(4, 24)
    dirty_vitals = noise_engine.apply_hadamard_mask(clean_vitals, drop_prob=0.25)
    mock_text_embeddings = torch.randn(4, 768)
    mock_targets = torch.tensor([1, 0, 1, 0], dtype=torch.long)

    # 3. Verify Forward Processing Pass
    model.zero_grad(set_to_none=True)
    outputs = model(vitals_features=dirty_vitals, text_features=mock_text_embeddings, labels=mock_targets)
    
    print(f"Forward Pass Success! Loss Output: {outputs['loss'].item():.4f}")
    
    # 4. Verify Post-Hoc Calibration Engine Logic
    calibration_engine = TemperatureCalibrationHead(model.temperature)
    calibration_engine.optimize_temperature(outputs['uncalibrated_logits'], mock_targets)
    print(f"Calibration Verified! Optimized global scalar value: {model.temperature.item():.4f}")