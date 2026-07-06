# push_to_hub.py
import os
from configuration_lemr import LEMRConfig
from modeling_lemr import LEMRForClinicalPrediction

if __name__ == "__main__":
    config = LEMRConfig(vitals_dim=24, text_dim=768, latent_dim=128)
    model = LEMRForClinicalPrediction(config)

    serialization_path = "./lemr-model-repo"
    os.makedirs(serialization_path, exist_ok=True)
    
    # Save code configurations locally
    model.save_pretrained(serialization_path)
    config.save_pretrained(serialization_path)

    # Push architecture blueprints to your global Hugging Face account
    # Change 'your-username' to your exact username profile string
    target_repo = "your-username/lemr-multimodal-clinical-core"
    
    print(f"Uploading files to repository space: {target_repo}...")
    model.push_to_hub(target_repo)
    config.push_to_hub(target_repo)
    print("Deployment finished successfully.")