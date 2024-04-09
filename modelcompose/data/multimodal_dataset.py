import os
import copy
import json
import glob
import random
import numpy as np
from PIL import Image
import torch
from collections import defaultdict
from dataclasses import dataclass
from torch.utils.data import Dataset
from typing import Dict, Optional, Sequence, List, Union

from modelcompose.data.utils import preprocess
from modelcompose.constants import MODAL_TOKENS, IGNORE_INDEX, VIDEO_CONFIG_PATH, MODEL_BASE

from .data import load_and_transform_audio_data, load_and_transform_video_data

def load_external_embeddings(extern_embedding_files: Union[str, List[str]], accumulate_idx=True):
    if isinstance(extern_embedding_files, str) and not os.path.exists(extern_embedding_files):
        extern_embedding_files = sorted(glob.glob(extern_embedding_files))
        
    if not isinstance(extern_embedding_files, str):
        embeddings = []
        idxs = []
        idx_key = None
        accumulate = 0
        for extern_embedding_file in extern_embedding_files:
            feats = torch.load(extern_embedding_file, map_location='cpu')
            embeddings.append(feats['embeddings'])
            if idx_key is None:
                if 'iids' in feats:
                    idx_key = 'iids'
                else:
                    idx_key = 'idxs'
            if idx_key == 'iids': 
                idxs.extend(feats[idx_key])
            elif idx_key == 'idxs': 
                if accumulate_idx:
                    for idx in feats[idx_key]:
                        idxs.append(idx + accumulate)
                else:
                    idxs.extend(feats[idx_key])
            accumulate = max(idxs) + 1
        return {'embeddings': torch.cat(embeddings), idx_key: idxs}
    else:
        return torch.load(extern_embedding_files, map_location='cpu')

class MultimodalDataset(Dataset):
    
    def __init__(
        self,
        data_path,
        tokenizer,
        data_args
    ):
        super().__init__()
        self.data_args = data_args
        self.tokenizer = tokenizer # tokenizer for LLM
        
        self.data = json.load(open(data_path))
        
        # HACK for video file check
        from modelcompose.model.multimodal_encoder.languagebind import LanguageBindVideoConfig
        from modelcompose.model.multimodal_encoder.languagebind import LanguageBindVideoProcessor
        video_config = LanguageBindVideoConfig.from_pretrained(VIDEO_CONFIG_PATH)
        self.video_processor = LanguageBindVideoProcessor(video_config)
        
    def __len__(self):
        return len(self.data)
    
    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.data:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            if len(sample.get('modal_inputs', {})) == 0:
                cur_len = -cur_len
            if 'vision' in sample.get('modal_inputs', {}):
                cur_len += 256
            if 'video' in sample.get('modal_inputs', {}):
                if sample['modal_inputs']['video'][0].endswith('.jpg'):
                    cur_len += 257
                else:
                    cur_len += 257 * 8
            # cur_len = cur_len if len(sample.get('modal_inputs', {})) != 0 else -cur_len
            length_list.append(cur_len)
        return length_list
    
    
    def get_modal_inputs(self, modal_inputs):
        for modal in modal_inputs:
            if modal == "vision":
                modal_inputs[modal] = [Image.open(img_fn).convert('RGB') for img_fn in modal_inputs[modal]]
                # modal_inputs[modal] = [Image.open(img_fn.replace('/yeesuanAI05/thumt/lfw/LLaVA/', '/yeesuanAI05/thumt/lfw/llava/LLaVA/')).convert('RGB') for img_fn in modal_inputs[modal]]
            elif modal == "audio":
                # modal_inputs[modal] = load_and_transform_audio_data(modal_inputs[modal], 'cpu') # self.data_args.device
                pass
            elif modal == "video":
                # modal_inputs[modal] = load_and_transform_video_data(modal_inputs[modal], 'cpu')
                
                # pass
                
                # HACK for video file check
                modal_inputs[modal] = self.video_processor(modal_inputs[modal])['pixel_values'] # 1, 3, 8, 224, 224
            
            
        return modal_inputs
    
    def __getitem__(self, index):
        example = copy.deepcopy(self.data[index])
        sources = [e["conversations"] for e in [example]]
        
        try:
            modal_inputs = self.get_modal_inputs(example.get('modal_inputs', {}))
        except:
            # There may be corrupted video files
            new_index = random.randint(0, len(self.data)-1)
            print(f"Corrupted: {index}, try {new_index}")
            return self.__getitem__(new_index)
        
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=len(modal_inputs) != 0)
        
        if isinstance(index, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
        data_dict['modal_inputs'] = modal_inputs
        return data_dict

class Dict2Class(object):
    
    def __init__(self, my_dict):
        
        for key in my_dict:
            setattr(self, key, my_dict[key])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: None
    modal_processors: None
    modal_configs: None # {'vision': {'image_aspect_ratio': 'pad'}}

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'modal_inputs' in instances[0]:
            batch['modal_inputs'] = defaultdict(list)
            
            # debug_output = "initial\n"

            for instance in instances:
                for modal_key, inputs in instance['modal_inputs'].items():
                    batch['modal_inputs'][modal_key].extend(inputs)
            #         # print(modal_key, type(inputs))
            #         debug_output += f"{modal_key} " + str(type(inputs)) + "\n"
            # debug_output += "finally " + str(type(batch['modal_inputs'])) + "\n"
            # print(debug_output)

            # exit(0) #
            batch['modal_inputs'] = self.process_modal_inputs(batch['modal_inputs'])

        return batch
    
    def process_modal_inputs(self, modal_inputs):
        results = dict()
        for key in modal_inputs:
            processor = self.modal_processors[key]
            if key == 'text':
                results[key] = processor(modal_inputs[key], return_tensors='pt', padding=True)
            elif key == 'vision':
                # results[key] = processor.preprocess(modal_inputs[key], return_tensors='pt')['pixel_values']
                from ..mm_utils import process_images
                if isinstance(self.modal_configs, dict) and 'vision' in self.modal_configs:
                    cfg = Dict2Class(self.modal_configs['vision'])
                results[key] = process_images(modal_inputs[key], processor, cfg)
            elif key == 'audio':
                audio_features, audio_padding_mask = processor(modal_inputs[key])
                results[key] = {
                    'audio_inputs': audio_features,
                    'audio_padding_mask': audio_padding_mask
                }
            elif key == 'video':
                # results[key] = modal_inputs[key]
                # results[key] = processor(modal_inputs[key])['pixel_values']
                
                # HACK for video file check
                num_frames = max([xx.shape[1] for xx in modal_inputs[key]])
                for i in range(len(modal_inputs[key])):
                    if modal_inputs[key][i].shape[1] != num_frames:
                        modal_inputs[key][i] = modal_inputs[key][i].expand(-1, num_frames, -1, -1)
                results[key] = torch.stack(modal_inputs[key], dim=0) # N, 3, 8, 224, 224
            elif key == 'point':
                results[key] = processor(modal_inputs[key]) # N, 8192, 6 (xyzrgb)
        return results
    
if __name__ == '__main__':
    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            MODEL_BASE,
            padding_side="right",
            use_fast=False,
        )
    
    class Dict2Class(object):
      
        def __init__(self, my_dict):
            
            for key in my_dict:
                setattr(self, key, my_dict[key])
    
    from modelcompose import conversation as conversation_lib
    
    # # Stage: Pretrain
    # data_args = {
    #     "is_multimodal": True
    # }
    # conversation_lib.default_conversation = conversation_lib.conv_templates["plain"]
    
    # Stage: Finetune
    data_args = {
        "is_multimodal": True,
    }
    conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    
    data_args = Dict2Class(data_args)
    
    dataset = MultimodalDataset(
        # "/yeesuanAI05/thumt/cc/MITv2/data/llava_lcs_558k/blip_laion_cc_sbu_558k_mm.json",
        "/yeesuanAI05/thumt/cc/MITv2/data/llava_intruct/llava_instruct_150k_mm_image.json",
        tokenizer,
        data_args=data_args
    )
    
    modal_processors={
        'text': transformers.CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14'),
        'vision': transformers.CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14')
    }
    collator = DataCollatorForSupervisedDataset(tokenizer, modal_processors)
    
    example = dataset[0]
    # import ipdb; ipdb.set_trace()
    batch = collator([dataset[i] for i in range(4)])
    

