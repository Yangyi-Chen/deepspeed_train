import os
import json
import yaml
import pathlib
import argparse
from tqdm import tqdm
import src.model
import src.data
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
from transformers import LlamaTokenizer, GenerationConfig
# from scripts.model.modeling_multimodal_mistral import MultimodalMistralForCausalLM
from transformers import MultimodalMistralForCausalLM
import wandb

def plot(loss_val):
    loss_values_np = np.array(loss_val)

    # Plot the loss curve
    plt.plot(loss_values_np)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.grid(True)

    # Save the figure to a file
    plt.savefig('loss_curve.png')

def main():
    # load config from args
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--local-rank", type=int,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")

    args = parser.parse_args()
    local_rank = args.local_rank
    # local_rank = os.environ['LOCAL_RANK']
    torch.distributed.init_process_group(backend="nccl")
    device = torch.device("cuda:{}".format(local_rank))



    with open(args.config, "r") as f:
        # safeload yaml
        config = yaml.safe_load(f)

    # import model and data
    # model = getattr(src.model, config["model"]["model_module"])(config)
    MODEL_PATH = "data/models/raw/model_iter_900"
    model = MultimodalMistralForCausalLM.from_pretrained("data/ckpt/SFT/final.pt", torch_dtype=torch.bfloat16)
    tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
    
    

    data_module = getattr(
        src.data, config["data"]["data_module"]
    )(config, tokenizer)
    max_epochs = config["training"]["max_epochs"]
    accumulate_grad_batches = config["training"]["accumulate_grad_batches"]
    checkpoint_every_n_steps = config["training"]["checkpoint_every_n_steps"]
    checkpoint_every_n_steps = checkpoint_every_n_steps * accumulate_grad_batches
    output_dir = config['output_dir']

    # checkpoint = config["model"].get("resume_from_checkpoint", None)

    # # checkpoint = None
    # if checkpoint is not None:
    #     print(f"Resuming training from {checkpoint}")
    #     # map_location = {"cuda:0": "cuda:{}".format(local_rank)}
    #     state_dict = torch.load(checkpoint, map_location='cpu')
    #     # print(state_dict.keys())
    #     new_dict = {}
    #     # for k, v in state_dict.items():
    #     #     if 'vision_language_model' in k or 'lora' in k or 'mlp' in k or 'embed_tokens' in k or 'input_layernorm' in k or "post_attention_layernorm" in k or 'lm_head' in k or 'norm' in k or 'linear' in k: 
    #     #         new_k = k
    #     #     else:
    #     #         new_k = k.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
    #     #     new_dict[new_k] = v
        
        
        
        
    #     # for k, v in state_dict.items():
    #     #     if 'language' == k[:8]:
    #     #         new_dict[k.replace("language_model.", 'language_model.base_model.model.')] = v
    #     #     else:
    #     #         new_dict[k] = v
    #     # state_dict = new_dict
        
    #     # new_dict = {}
    #     # for k, v in state_dict.items():
    #     #     if 'vision_language_model' in k or 'lora' in k or 'mlp' in k or 'embed_tokens' in k or 'input_layernorm' in k or "post_attention_layernorm" in k or 'lm_head' in k or 'norm' in k or 'linear' in k: 
    #     #         new_k = k
    #     #     else:
    #     #         new_k = k.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
    #     #     new_dict[new_k] = v
    #     # state_dict= new_dict
        
        
    #     msg = model.load_state_dict(state_dict, strict=False)
    #     print(msg)
    #     # print(state_dict.keys())
    #     # print("Load state dict: ", msg)
    #     # model.set_require_grad(mode=1)

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    # check the require_grad of model
    # for name, param in model.named_parameters():
    # print(name, param.requires_grad)

    data_loader = data_module.train_dataloader()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])
    step = config.get("init_step", 0)
    
    
    # lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 100,
                                                #    200000 // (accumulate_grad_batches))
    lr_scheduler =  get_cosine_schedule_with_warmup(optimizer, 100,
                                                   max_epochs * len(data_loader) // accumulate_grad_batches)
    lr = config["training"]["lr"]
    global_bs = config["data"]["batch_size"] * torch.cuda.device_count() * accumulate_grad_batches
    if local_rank == 0:
        wandb.init(
          # Set the project where this run will be logged
          project="mmistral-sft",
          name=f"lr{lr}-gbs{global_bs}"
          
        )
        
        
    
    loss_values = []
    all_loss_values = []

    try:
        for epoch in range(max_epochs):
            total_loss = 0.0
            model.train()
            tbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")
            for i, batch in enumerate(tbar):
                if batch is None:
                    print(f"Skipping batch {i} due to empty batch")
                    continue
                step += 1
                # if i < 122780:
                    # 82860:
                    # continue
                # if i == 82855:
                    # continue
                # if i in [2724, 15936, 15937, 15938, 15939, 18549, 25747, 25748, 25749, 28789, 78893]:
                #     continue
                
                
           
                
                loss = model(**batch)
                loss = loss.loss
                # continue if loss is nan
                
                if torch.isnan(loss):
                    print(f"Skipping batch {i} due to NaN loss, save")
                    record_step = step // accumulate_grad_batches
                    torch.save(model.module.state_dict(), os.path.join(output_dir, f"model_{record_step}.pt"))
                    continue
                
                    
                # clip gradient
                loss.backward()
                total_loss += loss.item()
                if args.local_rank == 0:
                    wandb.log({"loss": loss.item()})
                
                
                
                if step % accumulate_grad_batches == 0:
                    # average the gradients
                    for param in model.parameters():
                        if param.requires_grad:
                            if param.grad is None:
                                continue
                            param.grad.data /= accumulate_grad_batches
                            # except Exception:
                                # print(name)
                                # continue
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if step % 1000 == 0:
                    plot(loss_values)

                if step % checkpoint_every_n_steps == 0:
                    record_step = step // accumulate_grad_batches
                    if os.path.exists(output_dir):
                        pass
                    else:
                        os.makedirs(output_dir, exist_ok=True)
                    torch.save(model.module.state_dict(), os.path.join(output_dir, f"model_{record_step}.pt"))

            # avg_loss = total_loss / len(data_loader)
            # print(f"Epoch {epoch + 1}: Average Loss = {avg_loss:.2f}")
            plot(loss_values)
    except KeyboardInterrupt:
        pass
    plot(loss_values)
    if os.path.exists(output_dir):
        torch.save(model.module.state_dict(), os.path.join(output_dir, "model_final.pt"))
    else:
        os.makedirs(output_dir)
        torch.save(model.module.state_dict(), os.path.join(output_dir, "model_final.pt"))




if __name__ == '__main__':
    main()