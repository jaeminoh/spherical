from esm import pretrained
from peft import inject_adapter_in_model, LoraConfig

model_location = "checkpoints/esm2_t33_650M_UR50D.pt"
model, alphabet = pretrained.load_model_and_alphabet(model_location)

# how to inject LoRA adapters into the model
peft_config = LoraConfig(
    r=4, lora_alpha=1, bias="all",
    target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "fc1", "fc2", "dense"])    
model = inject_adapter_in_model(peft_config, model)

print(model)