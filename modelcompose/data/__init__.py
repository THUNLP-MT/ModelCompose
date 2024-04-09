from typing import Dict
import transformers
from .multimodal_dataset import MultimodalDataset, DataCollatorForSupervisedDataset

def make_multimodal_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                modal_data_configs) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = MultimodalDataset(
                                data_path=data_args.data_path,
                                tokenizer=tokenizer,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, modal_processors=data_args.modal_processors, modal_configs=modal_data_configs)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)