import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler
from datasets import load_dataset
from torch.utils.data import Dataset
from trl import PPOTrainer, PPOConfig
import numpy as np
import os
from tqdm import tqdm
from trl.models import AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from torch.nn.utils.rnn import pad_sequence
import time
import pdb

# Set environment variable to disable tokenizer parallelism warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Set environment variable for Hugging Face token
os.environ['HF_TOKEN'] = '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'

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description="Run PPO with transformers")
parser.add_argument("--n_updates", type=int, default=1000, help="Number of PPO updates")
parser.add_argument("--report_freq", type=int, default=1, help="Frequency of reporting metrics")
parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size for PPO training")
parser.add_argument("--learning_rate", type=float, default=1.41e-5, help="Learning rate for PPO training")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for PPO training")
parser.add_argument("--use_hf", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use Hugging Face API for generation")
parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for learning rate scheduler")
parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16"], help="Precision mode: fp32 or fp16")
parser.add_argument("--max_tokens", type=int, default=200)
args = parser.parse_args()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

mini_batch_size = args.batch_size

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name, peft_config=lora_config)

# Apply precision setting
if args.precision == "fp16":
    model = model.half()


class RandomTokenTruncateDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text
        tokenized = self.tokenizer.encode(text, return_tensors='pt').squeeze()
        # Choose a random point to truncate that includes at least one token
        truncate_point = random.randint(1, tokenized.size(0) - 1)  # Exclude the last token index to ensure it's not empty
        truncated_tokens = tokenized[:truncate_point]
        return truncated_tokens

# Load the dataset (example)
dataset = load_dataset("roneneldan/TinyStories", split='train')

# Preprocess the dataset to extract texts
texts = [example['text'] for example in tqdm(dataset.take(1000))]  # Taking only 1000 examples for simplicity

# Create the dataset using the tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')  # Use an appropriate tokenizer for your model
dataset = RandomTokenTruncateDataset(texts, tokenizer)


ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    mini_batch_size=mini_batch_size,
)

# Initialize the PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    config=ppo_config,
)

# Initialize the optimizer and learning rate scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=args.warmup_steps,
    num_training_steps=args.n_updates
)

generation_kwargs = {
    "min_length": -1,
    "max_new_tokens": args.max_new_tokens,
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
for n in tqdm(range(1, 1 + args.n_updates)):
    start = time.time()
    query_inds = np.random.randint(0, len(dataset), args.batch_size)
    queries = [dataset[i].cuda() for i in query_inds]
    print('max query length: ',  max([len(q) for q in queries]))

    if args.use_hf:
        flipped_queries = [q.flip(dims=(0,)) for q in queries]
        q_stack = pad_sequence(flipped_queries, batch_first=True, padding_value=tokenizer.pad_token_id)
        q_stack = torch.flip(q_stack, dims=[1])
        attn_mask = (q_stack != tokenizer.pad_token_id).long().cuda()

        with torch.no_grad():
            responses = model.generate(q_stack, attention_mask=attn_mask, max_new_tokens=args.max_new_tokens)  # Adjust max_length as needed
            responses = [seq[seq != tokenizer.pad_token_id] for seq in responses]

    else:
        responses = ppo_trainer.generate(queries, **generation_kwargs)

    rewards = compute_rewards(model, responses, queries)

    # Print the norm of the LoRA parameters
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
    lora_norm = torch.norm(torch.stack([torch.norm(p.detach()) for p in lora_params]))
    print(f"Step {n}, LoRA Params Norm: {lora_norm.item()}")

    for r in rewards:
        avg_reward += r.item()

    if n % args.report_freq == 0:
        print(avg_reward / (args.batch_size * args.report_freq), time.time() - start)
        avg_reward = 0

    # Perform PPO training step
    ppo_trainer.step(queries, responses, rewards)

    # Update the learning rate scheduler
    scheduler.step()

    # Clear CUDA memory
    torch.cuda.empty_cache()
