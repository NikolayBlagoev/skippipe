from simplellm.llama import SwapLLama, LLama
from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import RedPyjamav2, PretrainDataset
from sys import argv
import torch.distributed as dist
from torch import save, cuda, zeros_like, cat, mean, std
import torch
import traceback
import os
from simplellm.utils import State
import random
from torch.optim import AdamW
import json
from time import sleep
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from simplellm.losses import causalLLMLoss, perplexityLoss
random.seed(42)
State.set_seed(42)
torch.manual_seed(3407)
rank = int(argv[1])
world_size = int(argv[2])
skip = int(argv[3])
from simplellm.tokenizers.abstracttokenizer import AbstractTokenizer


class HFTokenizer(AbstractTokenizer):
    
    def __init__(self, tokenizer_name,access_token):
        self.tkns = AutoTokenizer.from_pretrained(tokenizer_name, token = access_token)
        self.vocab_size: int = 128256
        # print(self.tkns.bos_token)
        if self.tkns.bos_token != None:
            self.bos_id: int = self.tkns(self.tkns.bos_token).input_ids[0]
        else:
            self.bos_id = None
        self.eos_id = self.tkns(self.tkns.eos_token).input_ids[0]
        self.pad_id: int = 0
        
        

    def encode(self, txt: str) -> list[int]:
        return self.tkns(txt).input_ids

    def decode(self, tokens: list[int]) -> str:
        return self.tkns.decode(tokens)
    
    def batch_decode(self, *args, **kwargs):
        return self.tkns.batch_decode(*args, **kwargs)
device = f"cuda:{rank}"
dim = 1024
kv_heads = 16
layers = 24
stages = 6
layers_per_stage = layers // stages
ctx_size = 1024
lr = 4e-4
mb_c = 5
num_warmup_steps = 500
tokenizer = SPTokenizer()
padding_idx = tokenizer.eos_id
train_ds = RedPyjamav2(tokenizer, batch_size=8, skip = 1000, seq_l=ctx_size, name="sample-1T")
val_ds = RedPyjamav2(tokenizer, batch_size=8, skip = 0, seq_l=ctx_size, name="sample-1T")
configuration = LlamaConfig(hidden_size=dim,num_attention_heads=16,num_hidden_layers=24,max_position_embeddings=ctx_size,intermediate_size=dim*4, eos_token_id=padding_idx)
net = LlamaForCausalLM(configuration).to(device)
net.load_state_dict(torch.load("33-91/mdl.pth", weights_only=True))

with open("2_communication_8_samples_llama_500M.json","r") as fd:
    config = json.load(fd)
paths = config["ca-paths"]
partitions = config["partitions"]
sleep(rank*10)
tmp = {}
for idx, p in enumerate(partitions):
    for nd in p:
        tmp[nd] = idx
partitions = tmp
sizes = []
len_sizes = []
for param in net.parameters():
    sizes.append(param.shape)
    len_sizes.append(len(param.view(-1)))

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group("nccl", rank=rank, world_size=world_size)
tmp = []
for param in net.parameters():
    if param.data == None:
        tmp.append(torch.zeros_like(param,device=device).view(-1))
        continue
    tmp.append(param.data.view(-1))

tmp = cat(tmp)
print(tmp.shape)
dist.all_reduce(tmp, op = dist.ReduceOp.AVG)
tmp = torch.split(tmp, len_sizes)
# Sync model across devices...
for pi, param in enumerate(net.parameters()):
    param.data = tmp[pi].view(sizes[pi]).to(device)
del tmp
torch.cuda.empty_cache()
optimizer = AdamW(net.parameters(),lr = lr, betas=(0.9, 0.999), weight_decay=0)
# optimizer.load_state_dict(torch.load("optim.pth", weights_only=True))
train_dl = iter(train_ds)
print(net.dtype)
for itr in range(15_001):
    optimizer.zero_grad()
    if itr % 100 == 0 and rank == 0:
        net.eval()
        loss_hist = []
        with torch.no_grad():
            order = list(range(layers))
            val_dl = iter(val_ds)
            for _ in range(25):
                x = next(val_dl).to(device)
                target = x.detach().clone()
                x = net(x, order = order).logits
                # print(x)
                loss_hist.append(perplexityLoss(x,target).item())
            print(itr, "VALIDATION LOSS", sum(loss_hist)/len(loss_hist))
        save(net.state_dict(), "mdl.pth")
        save(optimizer.state_dict(), "optim.pth")
        net.train()

    loss_hist = 0
    for mb in range(mb_c):
        for k in range(world_size):
            try:
                if k == rank:
                    x = next(train_dl).to(device)
                else:
                    next(train_dl)
            except StopIteration:
                train_dl = iter(train_ds)
                if k == rank:
                    x = next(train_dl).to(device)
                else:
                    next(train_dl)
        target = x.detach().clone()
        if skip == 0 or itr % 10 == 0:
            order = list(range(layers))
            output = net(x, order = order).logits
        else:
            order = [kl for kl in range(layers_per_stage)]
            # print(mb,str(rank*3 + mb % 3))
            mb = paths[str(str(rank*3 + mb % 3))]
            
            for v in mb.values():
                order += list(range(layers_per_stage * partitions[v], layers_per_stage * (1 + partitions[v])))
            # print(mb)
            

            output = net(x, order = order).logits
        # print(output)
        # print(output.shape)
        print(output.dtype)
        loss = causalLLMLoss(output, target, tokenizer.vocab_size) / mb_c
        loss_hist += loss.item()
        loss.backward()
        del target
        del x
        torch.cuda.empty_cache()
    print(itr,"TRAINING LOSS", loss_hist)
    dist.barrier()
    tmp = []
    for param in net.parameters():
        if param.grad == None:
            tmp.append(torch.zeros_like(param,device=device).view(-1))
            continue
        tmp.append(param.grad.view(-1))

    tmp = cat(tmp)
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    dist.all_reduce(tmp, op = dist.ReduceOp.SUM)
    tmp = torch.split(tmp, len_sizes)
    # Sync model across devices...
    for pi, param in enumerate(net.parameters()):
        param.grad = tmp[pi].view(sizes[pi]).to(device)/world_size
    torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=1.0)
    optimizer.step()
    del tmp
    optimizer.zero_grad()
    torch.cuda.empty_cache()