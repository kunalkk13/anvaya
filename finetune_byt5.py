import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from datasets import load_dataset

# Define the tokenization function.
def tokenize_function(examples):
    input_texts = ["P2P " + s for s in examples["Sloka"]]
    target_texts = examples["Prose"]
    model_inputs = tokenizer(input_texts, max_length=512, truncation=True)
    labels = tokenizer(text_target=target_texts, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

model_name = "chronbmm/sanskrit5-multitask"  
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

raw_datasets = load_dataset("csv", data_files={"train": "/home/kunal/sanskrit/anvaya/data/rama_train_iast.csv", "test": "/home/kunal/sanskrit/anvaya/data/rama_test_iast.csv"})

# Tokenize the datasets and remove original columns.
tokenized_train = raw_datasets["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)
tokenized_test = raw_datasets["test"].map(
    tokenize_function,
    batched=True,
    remove_columns=raw_datasets["test"].column_names
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments.
training_args = TrainingArguments(
    output_dir="./byt5_sanskrit_full_finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",        # Evaluate at the end of each epoch.
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=2,
    logging_steps=100,
)

# Initialize the Trainer.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,  # If you want to use the test set for evaluation during training.
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Run full fine-tuning.
trainer.train()

# Save the fine-tuned model.
model.save_pretrained("./byt5_sanskrit_full_finetuned_final")
tokenizer.save_pretrained("./byt5_sanskrit_full_finetuned_final")
