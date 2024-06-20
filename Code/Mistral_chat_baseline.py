from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import torch

device = 'cpu'
if torch.cuda.is_available():
    device = torch.cuda.current_device()
elif torch.backends.mps.is_available():
    device = 'mps'

print(device)

token='***'

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", token=token)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", token=token)

test_df = pd.read_csv("TEST_headlines_23-24_Mar.csv", delimiter=';')
prompt = "Extract tickers of public companies mentioned in the headline: "

with open("outputs_mistral_baseline.txt", "w", encoding="utf-8") as f:
    for i in tqdm(range(len(test_df['headline'])), desc="Headlines Test Progress"):
        messages = [
            {"role": "user", "content": prompt + test_df['headline'].iloc[i]},
        ]

        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        f.write(decoded[0] + "\n")

