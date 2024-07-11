import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from trl import PPOTrainer, PPOConfig
import pdb
import os
from tqdm import tqdm
from trl.models import AutoModelForCausalLMWithValueHead
os.environ['HF_TOKEN'] = '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'

# Load the Tiny Stories dataset
dataset = load_dataset("roneneldan/TinyStories")

# Load the model and tokenizer
model_name = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)

# Preprocess the dataset
tokenized_data = []
for i, example in tqdm(enumerate(dataset['train'])):
    tokenized_data.append(tokenizer.encode(example['text'], return_tensors='pt').squeeze())
    if i == 1000: break

# Custom dataset class
class TensorDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx]

dataset = TensorDataset(tokenized_data)


ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.41e-5,
)


# Initialize the PPO trainer
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset,
    config=ppo_config
)


pdb.set_trace() 

# Custom reward function using model's own likelihood
def compute_rewards(input_ids):
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        rewards = log_probs.gather(2, input_ids.unsqueeze(-1)).squeeze(-1).mean(dim=1)
    return rewards

# Training loop
def training_step(batch):
    input_ids = batch.to(model.device)
    rewards = compute_rewards(input_ids)
    print(input_ids, rewards, flush=True)

    
    # Ensure rewards is a scalar tensor
    rewards_mean = rewards.mean().unsqueeze(0)
    
    # Perform PPO optimization step
    ppo_trainer.step(input_ids, rewards_mean)

    return rewards_mean.item()

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_dir='./logs',
    logging_steps=10,
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=lambda data: {'input_ids': torch.nn.utils.rnn.pad_sequence(data, batch_first=True)}
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained('./ppo_mistral_tinystories')
