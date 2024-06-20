from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import torch

device = 'cpu'
print(device)


model_name = "ceadar-ie/FinanceConnect-13B"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "right"

test_df = pd.read_csv("TEST_headlines_23-24_Mar.csv", delimiter=';')
prompt = "Extract tickers of public companies mentioned in the headline: "

with open("outputs_llama_finance.txt", "w", encoding="utf-8") as f:
    for i in tqdm(range(len(test_df['headline'])), desc="Headlines Test Progress"):
        messages = f"[INST] {prompt + test_df['headline'].iloc[i]} [/INST]"

        enc = tokenizer(messages, return_tensors="pt")

        model_inputs = enc.to(device)

        generated_ids = model.generate(input_ids=model_inputs.input_ids, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        f.write(decoded[0] + "\n")