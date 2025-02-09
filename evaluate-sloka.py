from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print('INFERENCE TIME')

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

FastLanguageModel.for_inference(model)

print('MODEL LOADED')

# Define the Alpaca prompt template
alpaca_prompt_inference = """
### INSTRUCTION:
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

### INPUT:
Sloka:
{sloka}

### RESPONSE:
"""





# Load the test data from CSV
test_data = pd.read_csv("test-anvaya-filtered.csv")

# Prepare the results list
results = []

# Define batch size
batch_size = 2

# Iterate over the test data in batches and generate outputs
for i in range(0, len(test_data), batch_size):
    print(i)
    batch = test_data.iloc[i:i + batch_size]
    inf_inputs = batch["Sloka"].tolist()
    samskrutas = batch["Prose"].tolist()
    
    # Prepare the inputs for the batch
    formatted_inputs = [
        alpaca_prompt_inference.format(
            inf_input,
            "",  # output - leave this blank for generation!
        ) for inf_input in inf_inputs
    ]
    
    inputs = tokenizer(formatted_inputs, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Generate the outputs
    outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
                             max_new_tokens=256, use_cache=True)
    generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # print(generated_texts)
    
    # Append the results
    for generated_text, samskruta, inf_input in zip(generated_texts, samskrutas, inf_inputs):
        results.append({"output": generated_text, "samskruta": samskruta, "english": inf_input})

# Create a DataFrame from the results
results_df = pd.DataFrame(results)

# Save the results to a new CSV file
results_df.to_csv("test-filtered-output", index=False)