from simplellm.tokenizers import SPTokenizer
from simplellm.dataloaders import Wikipedia_Dataset, PretrainDataset, FinetuneDataset, RedPyjama, Gutenberg_Project
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
from transformers import LlamaForCausalLM, LlamaConfig
from simplellm.losses import causalLLMLoss, perplexityLoss
random.seed(42)
State.set_seed(42)
torch.manual_seed(3407)

skip = int(argv[1])             # how inference should be performed
orig_model = int(argv[2])       # how was the original model trained
factor = 1
mdl = argv[3]
ds = argv[4]
device = f"cuda:{0}"

dim = 1024
kv_heads = 16
layers = 24
stages = 6 if orig_model != 25 else 4 # 25% model has 4 stages only
layers_per_stage = layers // stages
ctx_size = 1024
lr = 3e-4
mb_c = 5
num_warmup_steps = 500
tokenizer = SPTokenizer()
padding_idx = tokenizer.eos_id
if ds == "wikipedia":
    val_ds = Wikipedia_Dataset(tokenizer, batch_size=16, skip = 0, seq_l=ctx_size)
elif ds == "gutenberg":
    val_ds = Gutenberg_Project(tokenizer, split = "en", batch_size=16, skip = 0, seq_l=ctx_size)
elif ds == "stackexchange":
    val_ds = RedPyjama(tokenizer,group="stackexchange", batch_size=16, skip = 0, seq_l=ctx_size)
elif ds == "arxiv":
    val_ds = RedPyjama(tokenizer,group="arxiv", batch_size=16, skip = 0, seq_l=ctx_size)
configuration = LlamaConfig(hidden_size=dim,num_attention_heads=16,num_hidden_layers=24,max_position_embeddings=ctx_size,intermediate_size=dim*4, eos_token_id=padding_idx)
net = LlamaForCausalLM(configuration).to(device)
net.load_state_dict(torch.load(mdl, weights_only=True))

net.eval()
order = list(range(layers))
val_dl = iter(val_ds)
loss_hist = []
with torch.no_grad():
    for xi in range(50):
        for _ in range(xi):
            next(val_dl)
        x = next(val_dl).to(device)
        target = x.detach().clone()
        # print(order)
        post_factor = {}
        if skip == 0:
            order = list(range(layers))
            for k in order:
                if k < 4:
                    continue
                post_factor[k] = factor
            # output = net(x, order = order, post_layer_factor = post_factor).logits
        elif skip == 33 and orig_model != 25:
            order = list(range(stages - 1))
            order = random.sample(order,3)
            order.sort()
            
            tmp = [kl for kl in range(layers_per_stage)]
            for o in order:
                tmp += list(range(layers_per_stage * (1+o), layers_per_stage * (2+o)))
            order = tmp
        elif skip == 50 and orig_model != 25:
            order = list(range(stages - 1))
            order = random.sample(order,2)
            order.sort()
            
            tmp = [kl for kl in range(layers_per_stage)]
            for o in order:
                tmp += list(range(layers_per_stage * (1+o), layers_per_stage * (2+o)))
            order = tmp
        elif skip == 25 and orig_model == 25:
            order = list(range(stages - 1))
            order = random.sample(order,2)
            order.sort()
            
            tmp = [kl for kl in range(layers_per_stage)]
            for o in order:
                tmp += list(range(layers_per_stage * (1+o), layers_per_stage * (2+o)))
            order = tmp
        elif skip == 50 and orig_model == 25:
            order = list(range(stages - 1))
            order = random.sample(order,1)
            order.sort()
            
            tmp = [kl for kl in range(layers_per_stage)]
            for o in order:
                tmp += list(range(layers_per_stage * (1+o), layers_per_stage * (2+o)))
            order = tmp
            

        
        x = net(x, order = order, post_layer_factor = post_factor).logits            
        loss_hist.append(perplexityLoss(x,target).item())
print(sum(loss_hist) / len(loss_hist))

