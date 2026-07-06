# configuration_lemr.py
from transformers import PretrainedConfig

class LEMRConfig(PretrainedConfig):
    model_type = "lemr"
    def __init__(
        self, 
        vitals_dim: int = 24, 
        text_dim: int = 768, 
        latent_dim: int = 128, 
        num_classes: int = 2, 
        temperature: float = 1.0, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vitals_dim = vitals_dim
        self.text_dim = text_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.temperature = temperature