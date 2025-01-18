import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from peft import LoraConfig, get_peft_model

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

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

# Define the Alpaca prompt template
alpaca_prompt = """
### INSTRUCTION:
{}

### INPUT:
{}

### RESPONSE:
{}"""

# Instruction prompt describing the task
instruction_prompt = """ 
The goal is to convert the following Sanskrit verse to its respective Sanskrit prose. 
RULES
1. Sambodhya (vocative) comes at the initial position in the canonical order.
2. Kartṛ comes after vocative.
3. Kāraka relations follow in reverse order i.e. adhikaraṇa, apādāna, sampradāna, karaṇa and
karman.
4. Viśeṣanas, modifiers with genitive case markers, etc. are placed before their viśeṣya.
5. Kriyāviśeṣana, pratiṣedha etc. are placed right before their corresponding verb.
6. Mukhyakriyā is positioned at the end of the sentence.
7. Avyaya particles such as tu and api are placed right after their parent word.
8. The non-finite verbal forms are placed before the karman. All the arguments of non-finite
verb appear to their left.
9. The kartṛ-samānādhikaraṇa and karma-samānādhikaraṇa are placed after the katṛ and
karman respectively.
"""

EOS_TOKEN = tokenizer.eos_token

# Function to format the prompts
def formatting_prompts_func(examples):
    inputs = examples["Sloka"]
    outputs = examples["Prose"]
    texts = []
    for input_text, output in zip(inputs, outputs):
        text = alpaca_prompt.format(instruction_prompt, input_text, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

# Load the training dataset
df = pd.read_csv("train-seg-sloka.csv")
dataset = Dataset.from_pandas(df)

# Apply formatting to the dataset
dataset = dataset.map(formatting_prompts_func, batched=True)

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
print('STARTING TRAINING')
trainer_stats = trainer.train()

# Save the model and tokenizer
model.save_pretrained("seg-finetune-llama31")
tokenizer.save_pretrained("seg-finetune-llama31")