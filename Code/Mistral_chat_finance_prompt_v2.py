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

model = AutoModelForCausalLM.from_pretrained("briefai/LongShort-Mistral-7B", token=token)
tokenizer = AutoTokenizer.from_pretrained("briefai/LongShort-Mistral-7B", token=token)

test_df = pd.read_csv("TEST_headlines_23-24_Mar.csv", delimiter=';')
prompt = "Given the context, answer the question.\n \n### Question:\nExtract tickers of public companies mentioned in the headline.\n \n### Context:\n"

with open("outputs_mistral_finance_prompt_v2.txt", "w", encoding="utf-8") as f:
    for i in tqdm(range(len(test_df['headline'])), desc="Headlines Test Progress"):
        messages = [
            {"role": "user", "content": prompt + test_df['headline'].iloc[i]+ "\n \n### Answer:\n"},
        ]

        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")

        generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        f.write(decoded[0] + "\n")