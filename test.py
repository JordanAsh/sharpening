import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from trl import PPOTrainer, PPOConfig
import pdb
import numpy as np
import os
from tqdm import tqdm
from trl.models import AutoModelForCausalLMWithValueHead
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from torch.nn.utils.rnn import pad_sequence 

os.environ['HF_TOKEN'] = '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'

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

batch_size = 4 
mini_batch_size = 4

# Load the Tiny Stories dataset
dataset = load_dataset("roneneldan/TinyStories")

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, peft_config=lora_config)

# Custom dataset class
class TensorDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

# Preprocess the dataset
tokenized_data = []
for i, example in tqdm(enumerate(dataset['train'])):
    tokenized_data.append(tokenizer.encode(example['text'], return_tensors='pt').squeeze())
    if i == 1000: break
dataset = TensorDataset(tokenized_data)

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
    batch_size=batch_size,
    mini_batch_size=mini_batch_size,
)

# Initialize the PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    config=ppo_config,
)

generation_kwargs = {
    "min_length": -1,
    "max_length": 400, 
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}


# Custom reward function using model's own likelihood
def compute_rewards(model, input_ids, queries):
    inputs = [torch.cat([query, input_id]) for query,input_id in zip(queries, input_ids)] 
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.eos_token_id)
    with torch.no_grad():
        # inputs = torch.cat([queries.unsqueeze(0), input_ids],dim=1)
        outputs = model(padded_inputs, labels=padded_inputs)
        logits = [outputs[0][0, len(query)-1:len(query)-1+len(input_id), :] for query,input_id in zip(queries, input_ids)]
        # logits = outputs[0][0, query_lengths-1:-1, :]
        # rewards = logits[range(input_ids.shape[1]),input_ids[0]]
        rewards = [torch.mean(logits[i][range(len(input_id)), input_id]) for i, input_id in enumerate(input_ids)]
        # pdb.set_trace()
        # loss = torch.mean(rewards,dim=1)
    return rewards


n_updates = 1000
for n in range(n_updates):
    query_inds = np.random.randint(0, len(dataset), batch_size)
    queries = [dataset[i].to('cuda') for i in query_inds]
    # padded_queries = pad_sequence(queries, batch_first=True, padding_value=tokenizer.eos_token_id)
    # queries = padded_queries.to('cuda')
    responses = ppo_trainer.generate(queries, **generation_kwargs)
    rewards = compute_rewards(model, responses, queries)
    # avg_rewards = rewards.mean().unsqueeze(0)
    # pdb.set_trace()
    pdb.set_trace()
    ppo_trainer.step(queries, responses, rewards)



# # Define training arguments
# training_args = TrainingArguments(
#     output_dir='./results',
#     num_train_epochs=1,
#     per_device_train_batch_size=1,
#     logging_dir='./logs',
#     logging_steps=10,
# )

# # Create the Trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     data_collator=lambda data: {'input_ids': torch.nn.utils.rnn.pad_sequence(data, batch_first=True)}
# )


# # Train the model
# trainer.train()

# Save the model
# model.save_pretrained('./ppo_mistral_tinystories')
