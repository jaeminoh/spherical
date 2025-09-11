from transformers import AutoTokenizer, EsmForMaskedLM

output_dir = "checkpoints/pretrained"
# load model from Hugging Face
model_name = "facebook/esm2_t33_650M_UR50D"
model = EsmForMaskedLM.from_pretrained(model_name).save_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(model_name).save_pretrained(output_dir)
