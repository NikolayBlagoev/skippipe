from sys import argv
import torch.distributed as dist
from torch import save, cuda, zeros_like, cat, mean, std
import torch
import traceback
import os
from simplellm.tokenizers import SPTokenizer
from simplellm.utils import State
import random
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
import json
from time import sleep
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from simplellm.losses import causalLLMLoss, perplexityLoss2
import torch.nn.functional as F
random.seed(42)
State.set_seed(42)
torch.manual_seed(3407)
rank = int(argv[1])
world_size = int(argv[2])
skip = int(argv[3])

device = f"cuda:{rank}"
dim = 1024
kv_heads = 16
layers = 24
stages = 6
layers_per_stage = layers // stages
ctx_size = 1024
lr = 5e-6
mb_c = 2
num_warmup_steps = 500

mb_size = 8
tokenizer = SPTokenizer()
padding_idx = tokenizer.eos_id
def nex_el(tokenizer,dataset,current_iterator,keep=True):
    ret = []
    
    while len(ret) < mb_size:
        try:
            vl = next(current_iterator)
        except (StopIteration,Exception):
            current_iterator = iter(dataset)
            continue
        ret.append(vl)
    if not keep:
        return None, None, current_iterator
    ret_input_ids = []
    ret_targets = []
    for el in ret:
        # print("-----")
        el = el["messages"]
        loc_tmp_input_ids = []
        loc_tmp_target = []
        for t in el:
            loc_tmp_input_ids+=tokenizer("<|start_header_id|>" + t["role"][0] + "<|end_header_id|>" + t["content"][0]).input_ids
            # print(t["role"][0])
            if t["role"][0] == "assistant":
                loc_tmp_target+=tokenizer("<|start_header_id|>" + t["role"][0] + "<|end_header_id|>" + t["content"][0]).input_ids
            else:
                t = tokenizer("<|start_header_id|>" + t["role"][0] + "<|end_header_id|>" + t["content"][0]).input_ids
                loc_tmp_target+=[-100 for _ in t]
        ret_input_ids.append(torch.tensor(loc_tmp_input_ids))
        ret_targets.append(torch.tensor(loc_tmp_target))
    mx_size = 0
    for el in ret_input_ids:
        mx_size = max(mx_size,el.shape[0])
    mx_size = min(mx_size,1024)
    # print(mx_size)
    for el in range(len(ret_input_ids)):
        if ret_input_ids[el].shape[0] < mx_size:
            ret_input_ids[el] = F.pad(ret_input_ids[el], (0,mx_size - ret_input_ids[el].shape[0]), "constant", tokenizer.eos_token_id)
            ret_targets[el] = F.pad(ret_targets[el], (0,mx_size - ret_targets[el].shape[0]), "constant", -100)
        elif ret_input_ids[el].shape[0] > mx_size:
            ret_input_ids[el] = ret_input_ids[el][:mx_size]
            ret_targets[el] = ret_targets[el][:mx_size]
    


    # print(ret)
    return torch.stack(ret_input_ids), torch.stack(ret_targets), current_iterator
    

dataset = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True, trust_remote_code=True)
iterable_dataset = dataset.shuffle(buffer_size=50_000, seed=0).skip(1000)

ds = torch.utils.data.DataLoader(iterable_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=None, drop_last=True)
iterable_dataset = dataset.shuffle(buffer_size=50_000, seed=0).skip(0)
val_ds = torch.utils.data.DataLoader(iterable_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, collate_fn=None, drop_last=True)

net = LlamaForCausalLM.from_pretrained("")
net = net.to(device)
# net = LLama(SwapLLama,tokenizer.vocab_size, dmodel=dim, num_heads=kv_heads, n_layers=layers, ctx_size=ctx_size, device=device)
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
def causalLLMLoss(x, target, attention_mask = None, ignore_index=-100):
    x = x.float()
    target = target.to(x.device)
    target = F.pad(target, (0, 1), value=ignore_index)
    shift_labels = target[..., 1:].contiguous()
    shift_mask = None
    if attention_mask != None:
        
        shift_labels = shift_labels * attention_mask
    x = x.view(-1, x.size(-1))
    shift_labels = shift_labels.view(-1)
    loss = F.cross_entropy(x, shift_labels, ignore_index=ignore_index, reduction="mean")
    return loss

optimizer = AdamW(net.parameters(),lr = lr, betas=(0.9, 0.999), weight_decay=0)
# scheduler = LinearLR(optimizer, start_factor=0.03, total_iters=10_000)
train_dl = iter(ds)
epoch = 0
net.train()
for itr in range(5_001):
    optimizer.zero_grad()
    if itr % 25 == 0 and rank == 0:
        net.eval()
        loss_hist = []
        with torch.no_grad():
            order = list(range(layers))
            val_dl = iter(val_ds)
            for _ in range(25):
                x, target, val_dl = nex_el(tokenizer,val_ds,val_dl)
                
                input_ids = x.to(device)
                target = target.to(device)
                attention_mask = input_ids != tokenizer.eos_token_id
                action_mask = (target != -100) * attention_mask
                attention_mask = attention_mask.to(input_ids.dtype)
                action_mask = action_mask.to(input_ids.dtype)
                
                x = net(input_ids, attention_mask = attention_mask, order = order).logits
                loss = causalLLMLoss(x,target)
                if not loss.isfinite():
                    continue
                loss_hist.append(loss.item())
            print(itr, "VALIDATION LOSS", sum(loss_hist)/len(loss_hist))
        save(net.state_dict(), "mdl.pth")
        save(optimizer.state_dict(), "optim.pth")
        net.train()
    
    loss_hist = 0
    for mb in range(mb_c):
        k = 0
        while k < world_size:
            
            if k == rank:
                    x,target,train_dl = nex_el(tokenizer,ds,train_dl)
            else:
                    _,_, train_dl = nex_el(tokenizer,ds,train_dl,keep=False)
            k+=1
        
        input_ids = x.to(device)
        target = target.to(device)
        attention_mask = input_ids != tokenizer.eos_token_id
        attention_mask = attention_mask.to(input_ids.dtype)

        if skip == 0 or itr % 10 == 0:
            order = list(range(layers))
            output = net(input_ids, attention_mask = attention_mask, order = order).logits
        else:
            order = [kl for kl in range(layers_per_stage)]
            mb = paths[str(rank)]
            for v in mb.values():
                order += list(range(layers_per_stage * partitions[v], layers_per_stage * (1 + partitions[v])))
            # print(mb)
            # print(order)
            
            output = net(input_ids, attention_mask = attention_mask, order = order).logits

        
        loss = causalLLMLoss(output, target)
        if loss.isfinite():
            loss = loss/ mb_c
            loss_hist += loss.item()
            loss.backward()
        del input_ids
        del output
        del target
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
    dist.all_reduce(tmp, op = dist.ReduceOp.SUM)
    tmp = torch.split(tmp, len_sizes)
    # Sync model across devices...
    for pi, param in enumerate(net.parameters()):
        param.grad = tmp[pi].view(sizes[pi]).to(device)/world_size
    torch.nn.utils.clip_grad_norm_(net.parameters(),max_norm=1.0)
    optimizer.step()
    # scheduler.step()
    del tmp
    optimizer.zero_grad()
    torch.cuda.empty_cache()
    











