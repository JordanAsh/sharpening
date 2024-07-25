import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from trl import PPOTrainer, PPOConfig
import numpy as np
import os
from tqdm import tqdm
from trl.models import AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from torch.nn.utils.rnn import pad_sequence
import time
import deepspeed
import argparse
import bitsandbytes as bnb
from bitsandbytes.optim import Adam8bit

# Argument parser for precision level, batch size, mini batch size, max new tokens, and learning rate
parser = argparse.ArgumentParser(description='Train a model with specific configurations.')
parser.add_argument('--precision', type=str, choices=['fp16', 'bf16', 'int8'], default='fp16', help='Precision level for training (fp16, bf16, or int8).')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
parser.add_argument('--mini_batch_size', type=int, default=-1, help='Mini batch size for training.')
parser.add_argument('--max_new_tokens', type=int, default=400, help='Maximum number of new tokens to generate.')
parser.add_argument('--learning_rate', type=float, default=1.41e-5, help='Learning rate for the optimizer.')
parser.add_argument('--grad_steps', type=float, default=10000, help='')
parser.add_argument('--report_freq', type=float, default=5, help='')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()


n_updates = args.grad_steps
report_freq = args.report_freq
precision = args.precision
batch_size = args.batch_size
mini_batch_size = args.mini_batch_size
max_new_tokens = args.max_new_tokens
learning_rate = args.learning_rate
if mini_batch_size == -1: mini_batch_size = batch_size

os.environ['HF_TOKEN'] = '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'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load the Tiny Stories dataset
dataset = load_dataset("roneneldan/TinyStories")

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_size='left')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # Set padding side to left

if precision == 'int8':
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, peft_config=lora_config, load_in_8bit=True)
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, peft_config=lora_config)

model.config.pad_token_id = tokenizer.pad_token_id

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
    if args.debug and i > 1000: break
dataset = TensorDataset(tokenized_data)

# Custom collate function to pad sequences
def collate_fn(batch):
    batch = [item for item in batch]
    max_length = max([len(item) for item in batch])
    padded_batch = [torch.cat([torch.full((max_length - len(item),), tokenizer.pad_token_id), item]) for item in batch]
    return torch.stack(padded_batch, dim=0)

# Use DataLoader for efficient batching
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=learning_rate,
    batch_size=batch_size,
    mini_batch_size=mini_batch_size,
)

# Initialize DeepSpeed
ds_config = {
    "train_batch_size": batch_size,
    precision: {
        "enabled": True
    },
    "zero_optimization": {
        "stage": 2
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "gradient_accumulation_steps": 1,
    "steps_per_print": 100,
    "wall_clock_breakdown": False,
    "zero_allow_untested_optimizer": True  # Allow untested optimizer
}

# Custom optimizer class for integrating Adam8bit with DeepSpeed
class DeepSpeedAdam8bit(Adam8bit):
    def __init__(self, *args, **kwargs):
        super(DeepSpeedAdam8bit, self).__init__(*args, **kwargs)

# Initialize optimizer
if precision == 'int8':
    optimizer = DeepSpeedAdam8bit(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
else:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

# Initialize DeepSpeed with the model
model_engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=model.parameters(), optimizer=optimizer)

# Initialize the PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    config=ppo_config,
)

generation_kwargs = {
    "min_length": -1,
    "max_new_tokens": max_new_tokens,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}

# Custom reward function using model's own likelihood
def compute_rewards(model, input_ids, queries):
    inputs = [torch.cat([query, input_id]) for query, input_id in zip(queries, input_ids)]
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.eos_token_id)
    with torch.no_grad():
        outputs = model(padded_inputs, labels=padded_inputs)
        logits = [outputs[0][0, len(query)-1:len(query)-1+len(input_id), :] for query, input_id in zip(queries, input_ids)]
        rewards = [torch.mean(logits[i][range(len(input_id)), input_id]) for i, input_id in enumerate(input_ids)]
    return rewards


avg_reward = 0

for n in tqdm(range(1, 1 + n_updates)):
    start = time.time()
    queries = next(iter(data_loader))
    queries = torch.stack([q.cuda() for q in queries])
    attn_mask = (queries != tokenizer.pad_token_id).long().cuda()
    with torch.no_grad():
        responses = model_engine.generate(queries, attention_mask=attn_mask, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
        responses = [seq[seq != tokenizer.pad_token_id] for seq in responses]

    rewards = compute_rewards(model, responses, queries)
    for r in rewards: avg_reward += r.item()

    # Zero out gradients, compute gradients, and update weights
    model_engine.zero_grad()
    loss = -torch.mean(torch.stack(rewards).requires_grad_())
    model_engine.backward(loss)
    model_engine.step()

    if n % report_freq == 0:
        avg_reward = avg_reward / report_freq / batch_size
        print(f"Update {n}/{n_updates}, Average Reward: {avg_reward:.2f}, Time: {time.time() - start:.2f}s")

