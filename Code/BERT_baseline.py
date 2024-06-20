import pandas as pd
from transformers import BertForMaskedLM, BertTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import torch

device = torch.device("mps") if torch.has_mps else torch.device("cpu")

def format_example(row):
    headline = row['headline']
    tickers = row['ticker']
    return f"{headline}. The tickers of public companies mentioned in the headline are {tickers}"


def tokenize_function(examples):
    return tokenizer(examples["formatted"], padding="max_length", truncation=True)


train_df = pd.read_csv("TRAIN_headlines_23-24_Mar.csv", delimiter=';')
train_df["formatted"] = train_df.apply(format_example, axis=1)
dataset = Dataset.from_pandas(train_df[["formatted"]])


model_checkpoint = "google-bert/bert-base-uncased"
model = BertForMaskedLM.from_pretrained(model_checkpoint)
model.to(device)

tokenizer = BertTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["formatted"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets,
    data_collator=data_collator,
)


trainer.train()
trainer.save_model("fine-tuned-bert")


test_df = pd.read_csv("TEST_headlines_23-24_Mar.csv", delimiter=';')

print('TESTING ON TEST SET')

for item in range(len(test_df['headline'])):
    input_text = test_df['headline'].iloc[item] + ". The tickers of public companies mentioned in the headline are [MASK] [MASK]"

    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

    predicted_token_ids = logits[0, mask_token_index].argmax(axis=-1)
    generated_tickers = tokenizer.decode(predicted_token_ids)

    print("Headline: ", test_df['headline'].iloc[item])
    print("Generated Tickers:", generated_tickers)