import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model and tokenizer configuration
max_seq_length = 2048
load_in_4bit = True  # Reduces memory usage
dtype = None  # Auto-detect precision

# Load the base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Load Formatted dataset
df = pd.read_csv("formatted_train_anvaya.csv")
dataset = Dataset.from_pandas(df)


# Fine-tuning configuration
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=5,
        output_dir="outputs",
        save_strategy="epoch",
    ),
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained("fine_tuned_llama3")
tokenizer.save_pretrained("fine_tuned_llama3")