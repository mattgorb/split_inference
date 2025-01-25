import torch
import torch.nn as nn
import os
import torch.optim as optim
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
import time
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch
from transformers import AutoConfig, AutoTokenizer,AutoModelForCausalLM
import torch.nn as nn
import logging
logging.basicConfig(level=logging.DEBUG)


"""
export MASTER_ADDR="129.82.45.26"  # Master Node IP Address for teal

module purge
module load python/anaconda
"""


class GPT2(torch.nn.Module):
    def __init__(self, config,offset_layers=None):

        super(GPT2, self).__init__()
        self.config=config
        self.default_config = AutoConfig.from_pretrained(config, cache_dir='model_store')
        hf_model = AutoModelForCausalLM.from_pretrained(config, cache_dir='model_store' )

        self.transformer=hf_model.transformer
        self.transformer.h= torch.nn.ModuleList(hf_model.transformer.h[offset_layers:])

        self.transformer.ln_f=hf_model.transformer.ln_f
        del hf_model
        del self.transformer.wte
        del self.transformer.wpe
        del self.transformer.drop
        torch.cuda.empty_cache()

    def forward(self, hidden_states,  **kwargs):
        for _, block in enumerate(self.transformer.h):
            hidden_states = block(hidden_states, )[0]
        hidden_states=self.transformer.ln_f(hidden_states)
        return hidden_states




def generate(model_owner,device,  rank, group_mo, group_all, model_owner_initial_rank=1):
    curr_batch_size=torch.zeros(1).int().to(device)
    curr_context_size=torch.zeros(1).int().to(device)
    while True:

        #at the end of training and test loops, the batch size will sometimes be smaller (since there arent many samples left. we need to communicate this. )
        if rank==model_owner_initial_rank:
            dist.recv(curr_batch_size, src=0,)
        dist.broadcast(curr_batch_size, src=model_owner_initial_rank, group=group_mo)
        #sending 0 from the data owner to simulate end of training.
        if curr_batch_size==0:
            break

        if rank==model_owner_initial_rank:
            dist.recv(curr_context_size, src=0,)
        dist.broadcast(curr_context_size, src=model_owner_initial_rank, group=group_mo)
        dist.barrier(group_all)


        data_owner_intermediate_output=nn.Parameter(torch.zeros(curr_batch_size[0],curr_context_size[0],model_owner.default_config.hidden_size ), requires_grad=True).to(device)
        if rank==model_owner_initial_rank:
            dist.recv(data_owner_intermediate_output, src=0,)
        dist.broadcast(data_owner_intermediate_output, src=model_owner_initial_rank,group=group_mo)
        dist.barrier(group_all)

        data_owner_intermediate_output.retain_grad()
        model_owner_output = model_owner(data_owner_intermediate_output, )
        if rank==model_owner_initial_rank:
            dist.send(model_owner_output, dst=0)

        dist.barrier(group_all)
        dist.barrier(group_all)


# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_owner_total_procs', type=int, default=1 )
    parser.add_argument('--model_owner_total_procs', type=int, default=1 )
    parser.add_argument('--communication_rounds', type=int, default=1, )
    
    parser.add_argument('--model_name', type=str, default='gpt2', )

    parser.add_argument('--rank', type=int, default=1, )
    parser.add_argument('--local_rank', type=int, default=1, )
    parser.add_argument('--world_size', type=int, default=2, )

    parser.add_argument('--master_address', type=str, default="192.168.1.153", )
    parser.add_argument('--master_port', type=str, default="29500", )
    parser.add_argument('--device', type=str, default="cuda", )
    parser.add_argument('--ifname', type=str, default="en0", )
    
    args = parser.parse_args()

    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    os.environ["RANK"] = str(args.rank)


    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    
    data_owner_total_procs=args.data_owner_total_procs
    model_owner_initial_rank=data_owner_total_procs

    print(f'distributed setup: rank: {rank},   local_rank: {local_rank}, world size: {world_size}, master_address: {args.master_address}, master_port: {args.master_port} ')

    if args.device=='cuda' and torch.cuda.is_available():
        print('starting nccl backend (GPU)')
        os.environ['NCCL_SOCKET_IFNAME']=str(args.ifname)
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl',  init_method=f'env://')
        device = torch.device(f"cuda:{local_rank}")
    else:
        print('starting gloo backend (CPU)')
        os.environ['GLOO_SOCKET_IFNAME']=str(args.ifname)
        dist.init_process_group(backend='gloo',  init_method=f'env://')
        device='cpu'

    print(f'distributed environment created successfully with device: {device}')
    model_owner_ranks=[i for i in range(model_owner_initial_rank, model_owner_initial_rank+args.model_owner_total_procs)]
    all_ranks=[i for i in range(args.data_owner_total_procs+args.model_owner_total_procs)]

    print(f'model owner ranks: {model_owner_ranks}')
    print(f'all_ranks ranks: {all_ranks}')
    group_all=dist.new_group(ranks=all_ranks)
    group_mo = dist.new_group(ranks=model_owner_ranks)
    dist.barrier(group=group_all)


    print(f"Process group initialized for rank {rank}, local rank: {local_rank} world size {world_size}.")
    model_owner = GPT2(args.model_name, offset_layers=1).to(device)

    #print("Converting model to FSDP")
    dist.barrier(group=group_mo)
    #model_owner=FSDP(model_owner, process_group=group_mo)
    model_owner.eval()

    dist.barrier(group=group_all)

    #dist.barrier(group=group_all)
    generate(model_owner, device, rank, group_mo, group_all ,model_owner_initial_rank=1)
    

if __name__ == "__main__":
    main()







