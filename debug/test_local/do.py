'''
#Simple MNIST example.  
#Run these before executing scripts
#to get IP: ifconfig, inet (first value)



export MASTER_ADDR="192.168.1.153"  # Master Node IP Address for LOCAL OHIO
export MASTER_PORT="29500"           # Port for communication
export NCCL_SOCKET_IFNAME=en0

lsof -ti :29500 | xargs -r kill -9


export NCCL_DEBUG=INFO
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1  



#do.py
export MASTER_ADDR="192.168.1.153"
export MASTER_PORT="29500"
export WORLD_SIZE="2"
export LOCAL_RANK="0"
export RANK="0"



#mo.py
export MASTER_ADDR="192.168.1.153"
export MASTER_PORT="29500"
export WORLD_SIZE="2"
export LOCAL_RANK="0"
export RANK="1"


module purge
module load python/anaconda
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
#from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import argparse
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from torch.nn import functional as F

from transformers import GPT2LMHeadModel
#from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoConfig, AutoTokenizer,AutoModel, AutoModelForCausalLM

class GPT2LMHEAD(GPT2LMHeadModel):
    def __init__(self, config, num_layers):
        super().__init__(config)
        self.config.output_hidden_states = True
        self.transformer.h=self.transformer.h[:num_layers]
        self.transformer.ln_f = torch.nn.Identity()
        self.lm_head=torch.nn.Identity()
    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        return outputs.hidden_states[-1]
    
class GPT2LMTAIL(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        full_model = AutoModelForCausalLM.from_pretrained(config, cache_dir='model_store')
        self.lm_head = full_model.lm_head  # Copy only the lm_head
        del full_model  # Free memory by deleting the full model
    def forward(self, hidden_states):
        logits = self.lm_head(hidden_states)
        return logits





def model_communication(model_head, model_tail, tokens, group_all, device):
    dist.send(torch.tensor([tokens['input_ids'].size(0)]).int().to(device), dst=1)
    dist.send(torch.tensor([tokens['input_ids'].size(1)]).int().to(device), dst=1)
    dist.barrier(group=group_all) #1

    hidden_states = model_head(**tokens)

    dist.send(hidden_states, dst=1)
    dist.barrier(group=group_all) #2

    hidden_states=torch.zeros_like(hidden_states).to(device)

    #receive_from_any_known_sources(flag, data_owner_ranks)
    dist.recv(hidden_states, src=1)
    dist.barrier(group=group_all) #3

    output = model_tail(hidden_states)
    
    dist.barrier(group_all)

    return output

@torch.no_grad()
def generate(model_head, model_tail, group_all, device, tokens, max_new_tokens=64, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        logits = model_communication(model_head, model_tail, tokens, group_all, device)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution
        tokens_next = torch.multinomial(probs, num_samples=1)
        #tokens_next = torch.argmax(logits, dim=-1, keepdim=True)

        # append sampled index to the running sequence and continue
        tokens['input_ids'] = torch.cat((tokens['input_ids'], tokens_next), dim=1)
        #print(tokens['input_ids'].size())
        tokens['attention_mask'] = torch.cat(
            (tokens['attention_mask'], torch.ones((tokens['input_ids'].shape[0], 1), dtype=torch.long, device=device)), 
            dim=1
        )
        #print(tokens['attention_mask'])
    return tokens


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_owner_total_procs', type=int, default=1,)
    parser.add_argument('--model_owner_total_procs', type=int, default=1,)
    parser.add_argument('--communication_rounds', type=int, default=1,)
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = "192.168.1.153"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["RANK"] = "0"

    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    if torch.cuda.is_available():
        print('starting nccl backend (GPU)')
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend='nccl',  init_method=f'env://')
    else:
        print('starting gloo backend (CPU)')
        dist.init_process_group(backend='gloo',  init_method=f'env://')
        device='cpu'

    #dist.init_process_group(backend='nccl',  init_method=f'env://')
    #dist.init_process_group(backend='gloo',  init_method=f'env://')

    data_owner_ranks=[i for i in range(args.data_owner_total_procs)]
    all_ranks=[i for i in range(args.data_owner_total_procs+args.model_owner_total_procs)]

    print(f'data owner ranks: {data_owner_ranks}')
    print(f'all_ranks ranks: {all_ranks}')

    group_all=dist.new_group(ranks=all_ranks)
    group_do = dist.new_group(ranks=data_owner_ranks)

    dist.barrier(group=group_all)
    
    print(f"Process group initialized for rank {rank}, local rank: {local_rank} world size {world_size}.")
    print(f"Rank {dist.get_rank()} is part of group_do with ranks {data_owner_ranks}")
    
    
    # Tokenize input
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # Add a padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    input_text = [
                   "Hello, how are you today?", 
                   #"What is the weather like?",
                   #"I like traveling by train because",
                   #"Artificial Intelligence (AI) has seen tremendous growth over the last decade, leading to advancements in natural language processing, computer vision, and robotics. What do you think the future of AI holds?", 
                ]
    tokens = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

    model_head = GPT2LMHEAD.from_pretrained("gpt2", num_layers=1).to(device)
    model_tail = GPT2LMTAIL("gpt2", ).to(device)
    model_head.eval()
    model_tail.eval()

    dist.barrier(group=group_all)

    tokens_out=generate(model_head, model_tail, group_all, device, tokens, max_new_tokens=64, temperature=1.0, top_k=None)


    decoded_texts = [
        tokenizer.decode(seq.tolist(), skip_special_tokens=True)
        for seq in tokens_out['input_ids']
    ]

    # Display results
    for i, text in enumerate(decoded_texts):
        print(f"Generated Text {i + 1}:\n{text}\n")

    dist.send(torch.zeros(1).to(device), dst=1)

if __name__ == "__main__":
    main()


