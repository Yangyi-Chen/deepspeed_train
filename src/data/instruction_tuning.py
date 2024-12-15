import os
import torch
import json
import numpy as np
import jsonlines
from tqdm import tqdm
from src.data.utils import (
    load_image_to_base64,
    convert_image_base64_to_patches,
)




IGNORE_INDEX = -100
def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data
def read_jsonlines(file_path):
    with jsonlines.open(file_path) as reader:
        data = [obj for obj in reader]
    return data

B_INST, E_INST = "[INST]", "[/INST]"


class SFTModule():
    def prepare_inputs_img(self, images, inputs, tokenizer):
        end_token = tokenizer.eos_token
        
        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []
        
        patches = images
        n_rows, n_cols = patches.shape[:2]
        n_patches = n_rows * n_cols
        patches = patches.view(n_patches, -1)
        
        # ---
        img_tokens = ["<vision>"]
        cur_patch_indices = [NON_VISION_TOKEN]
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                if row_idx != 0 and col_idx == 0: # when new row starts
                    img_tokens.append(f"<vrow_sep>")
                    cur_patch_indices.append(NON_VISION_TOKEN)
                img_tokens.append(f"<vpatch>")
                cur_patch_indices.append(len(vision_patches) + row_idx * n_cols + col_idx)
        img_tokens.append("<vision>")
        cur_patch_indices.append(NON_VISION_TOKEN)
        
        # ---
        # NOTE tokenizer(xxx) will NOT work here
        cur_tokens = torch.Tensor(tokenizer.convert_tokens_to_ids(img_tokens))
        cur_attention_mask = [1] * len(cur_tokens)
        # print(f"cur_tokens: {cur_tokens}")
        # print(f"cur_attention_mask: {cur_attention_mask}")
        # print(f"cur_patch_indices: {cur_patch_indices}")
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"
        
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens)) 
        attention_masks.extend(cur_attention_mask)
        vision_patch_indices.extend(cur_patch_indices)
        vision_patches.extend(patches.numpy().astype(np.float16))
        
        
        
        
        for idx, i in enumerate(inputs):
            if idx % 2 == 0:
                if idx == 0:
                    i = i.replace("<image>\n", '').replace("\n<image>", '')
                    c_new = tokenizer.bos_token + f"{B_INST} {i.strip()} {E_INST}"                                       
                else:
                    c_new = f"{B_INST} {i.strip()} {E_INST}"
                _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
                cur_tokens = _tokenized["input_ids"].squeeze(0)
                cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
                tokens.extend(cur_tokens)
                labels.extend([-100] * len(cur_tokens))
                attention_masks.extend(cur_attention_mask)
                vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
            else:
                i = i + end_token
                _tokenized = tokenizer(i, return_tensors="pt", add_special_tokens=False)
                cur_tokens = _tokenized["input_ids"].squeeze(0)
                cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
                tokens.extend(cur_tokens)
                labels.extend(cur_tokens)
                attention_masks.extend(cur_attention_mask)
                vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        
        
        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)
            
            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels



    def prepare_inputs(self, inputs, tokenizer):
        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []
        
        
        for idx, i in enumerate(inputs):
            if idx % 2 == 0:
                if idx == 0:
                    c_new = tokenizer.bos_token + f"{B_INST} {i.strip()} {E_INST}"                                       
                else:
                    c_new = f"{B_INST} {i.strip()} {E_INST}"
                _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
                cur_tokens = _tokenized["input_ids"].squeeze(0)
                cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
                tokens.extend(cur_tokens)
                labels.extend([-100] * len(cur_tokens))
                attention_masks.extend(cur_attention_mask)
                vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
            else:
                i = i + tokenizer.eos_token
                _tokenized = tokenizer(i, return_tensors="pt", add_special_tokens=False)
                cur_tokens = _tokenized["input_ids"].squeeze(0)
                cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
                tokens.extend(cur_tokens)
                labels.extend(cur_tokens)
                attention_masks.extend(cur_attention_mask)
                vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        
        
        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
        
        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        
        return tokens, attention_masks, vision_patches, vision_patch_indices, labels
    
            
        
        
        



        
    
    def collate_fn(self, batch):
        try:
            assert len(batch) == 1
            for i, tgt_item in enumerate(batch):
                conversation_li = []
                conversations = tgt_item['conversations']
                for item in conversations:
                    if type(item) is str:
                        conversation_li.append(item)
                    else:
                        conversation_li.append(item['value'])
                if 'image' in tgt_item:
                    orig_img_path = tgt_item['image']
                    img_base64 = load_image_to_base64(orig_img_path)
                    img_patches = convert_image_base64_to_patches(img_base64)
                    tokens, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs_img(img_patches, conversation_li, self.tokenizer)
                else:
                    tokens, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs(conversation_li, 0, self.tokenizer)

            return {
                "input_ids":tokens.unsqueeze(0),
                "attention_mask":attention_masks.unsqueeze(0),
                "vision_patches":vision_patches,
                "vision_patch_indices":vision_patch_indices.unsqueeze(0),
                "labels": labels.unsqueeze(0)
            }
                
        
        except Exception as e:
            print(e)
            return None
    
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.all_data,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
    
        
    
    def __init__(self, config: dict, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_position_embeddings = config["data"]["max_position_embeddings"]
         

        self.visual_data = read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/the_cauldron/all_data.jsonl")
        self.language_data = read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/language_only/language_sft.jsonl") + read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/tulu_language_sft/tulu-3-sft-mixture.jsonl")
        
        self.sharegpt4v = read_jsonlines('/shared/nas/data/m1/yangyic3/MultimodalAgent/data/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_filter.jsonl')
        self.gpt4laion = read_jsonlines("/shared/nas/data/m1/yangyic3/MultimodalAgent/data/gpt4laion/gpt4laion_filter.jsonl") # yes
        self.lvis_instruct = read_jsonlines("/shared/nas/data/m1/yangyic3/MultimodalAgent/data/LVIS-Instruct4V/lvis_instruct4v_220k.jsonl")
        self.gqa = read_jsonlines("/shared/nas/data/m1/yangyic3/MultimodalAgent/data/gqa.jsonl")
        self.textocr_gpt4v = read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/textocr/textocr-gpt4v/textocr_gpt4v.jsonl")
        self.llavar = read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/LLaVAR/chat_llavar_filtered.jsonl")
        self.pixmo_askanything = read_jsonlines("/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/pixmo/pixmo.jsonl") 
        
        
        
        for item in self.visual_data:
            item['image'] = "/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/the_cauldron/images/" + item['image']
        for item in self.language_data:
            item['source'] = "language_only"
        for item in self.sharegpt4v:
            item['image'] = "/shared/nas/data/m1/yangyic3/MultimodalAgent/data/" + item['image']
            item['source'] = "sharegpt4v"
        for item in self.gpt4laion:
            item['image'] = "/shared/nas/data/m1/yangyic3/MultimodalAgent/data/" + item['image']
            item['source'] = "gpt4laion"
        for item in self.lvis_instruct:
            item['image'] = "/shared/nas/data/m1/yangyic3/MultimodalAgent/data/" + item['image']
            item['source'] = "lvis_instruct"
        for item in self.gqa:
            item['image'] = "/shared/nas/data/m1/yangyic3/MultimodalAgent/data/" + item['image']
            item['source'] = "gqa"
        for item in self.textocr_gpt4v:
            item['source'] = "textocr_gpt4v"
        for item in self.pixmo_askanything:
            item['image'] = "/shared/nas/data/m1/yangyic3/Multimodal-Mistral/data/raw_data/" + item['image']
            item['source'] = "pixmo_askanything"
        
        
        
        self.all_data = self.visual_data + self.language_data + self.gqa + self.textocr_gpt4v + self.sharegpt4v + self.gpt4laion + self.lvis_instruct + self.pixmo_askanything
        #+ self.llavar+ self.allava_laion + self.allava_vflan 
        print("TOTAL data: ", len(self.all_data))


