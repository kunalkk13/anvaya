import pandas as pd
import torch
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

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

# Define the inference Alpaca-style prompt (without dependency table)
alpaca_prompt_inference = """
### INSTRUCTION:
The goal is to convert the following Sanskrit verse to its respective Sanskrit prose. Use the following rules to convert the verse to its respective prose.
RULES :
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
# Load the test dataset (assumed to have a "Sloka" column)
test_df = pd.read_csv("test_anvaya.csv")

# Define batch size (adjust based on available GPU memory)
batch_size = 2  

# Prepare results list
results = []

# Process test dataset in batches
for i in range(0, len(test_df), batch_size):
    print(f"Processing batch {i}/{len(test_df)}")
    
    # Get batch
    batch = test_df.iloc[i : i + batch_size]
    slokas = batch["Sloka"].tolist()
    
    # Format input prompts for each Sloka
    formatted_inputs = [alpaca_prompt_inference.format(sloka=sloka) for sloka in slokas]
    
    # Tokenize inputs
    inputs = tokenizer(formatted_inputs, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate outputs
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=256
    )
    
    # Decode outputs to get generated prose
    generated_proses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Append results
    for sloka, prose in zip(slokas, generated_proses):
        results.append({"Sloka": sloka, "Generated_Prose": prose})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV for evaluation
results_df.to_csv("anvaya_test_outputs.csv", index=False)

print("Inference completed! Results saved to 'anvaya_test_outputs.csv'.")