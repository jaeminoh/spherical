from transformers import AutoTokenizer, EsmForMaskedLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import peft
from Bio import SeqIO


# load model from Hugging Face
model_name = "checkpoints/pretrained"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForMaskedLM.from_pretrained(model_name)


class FastaDataset(Dataset):
    def __init__(self, path: str, tokenizer: AutoTokenizer):
        self.sequences = []
        for seq_record in SeqIO.parse(path, "fasta"):
            self.sequences.append(
                tokenizer(str(seq_record.seq), return_tensors="pt")["input_ids"].squeeze()
            )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx: int):
        return self.sequences[idx]

dataset = FastaDataset("assets/Phenuiviridae/Phenuiviridae.fasta", tokenizer)
test = FastaDataset("assets/Phenuiviridae/Phlebovirus.fasta", tokenizer)

# LoRA adapter injection
peft_config = peft.LoraConfig(
    r=4, lora_alpha=1, bias="all",
    target_modules=["query", "key", "value", "dense"]) # warning: different from esm github.

model = peft.get_peft_model(model, peft_config)
print(model.print_trainable_parameters()) # trainable params: 3,520,034 || all params: 654,094,774 || trainable%: 0.5382


output_dir = "checkpoints/finetuned"
training_args = TrainingArguments(
    output_dir=output_dir,
    eval_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=100,
)

data_collator = DataCollatorForLanguageModeling(tokenizer, pad_to_multiple_of=256)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(f"{output_dir}/final")
